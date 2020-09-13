import os
import pandas as pd
from typing import List, Optional, Tuple

import boto3
import argparse
from pprint import pprint
from glob import glob

from src.comprehend_utils import Sentiment, analyze_sentiment
from src.transcribe_utils import Speech, TranscriptReport, download_transcripts, get_recording_files, move_transcripts, \
    report_transcript, \
    transcribe_recording, \
    wait_for_jobs


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bucket", type=str, default="zrmar-transcribe-jobs")
    parser.add_argument("--source-path", type=str, default="audio/contact-center/16KHz")
    parser.add_argument("--dest-bucket", type=str, default="zrmar-transcribe-jobs")
    parser.add_argument("--dest-path", type=str, default="transcribe-output")
    parser.add_argument("--local-path", type=str, default="../data/transcriptions")
    parser.add_argument("--local-df-path", type=str, default="../data/results/transcription")
    parser.add_argument("--base-job-name", type=str, default="dpv-cc-v2")
    parser.add_argument("--operation", type=str, default="transcribe",
                        choices=["transcribe", "download", "process"],
                        help="Set the operation to execute: transcribe, download or all")
    return parser.parse_args()


def transcribe_recordings(args: argparse.Namespace) -> None:
    base_job_name = args.base_job_name
    bucket = args.source_bucket
    path = args.source_path
    file_names = get_recording_files()
    print(file_names)
    transcribe = boto3.client('transcribe')
    transcribe_jobs = []
    for i, recording in enumerate(file_names):
        transcribe_jobs.append(transcribe_recording(recording, f"{base_job_name}-{i + 1:03d}"))
    final_job_results = wait_for_jobs(transcribe_jobs)
    move_transcripts(jobs=final_job_results, dest_bucket=args.dest_bucket, dest_path=args.dest_path)
    pprint(final_job_results)


def process_transcripts(
    args: argparse.Namespace, mask: str = "*.json"
) -> List[Tuple[str, List[Speech], List[Sentiment]]]:
    transcribe = boto3.client('transcribe')
    comprehend = boto3.client('comprehend')
    dialogues = []
    for transcript in glob(f"{args.source_path}/{mask}"):
        transcript_report = report_transcript(transcript)
        general_sentiment = analyze_sentiment(transcript_report.full_text[:4500])
        sentiments = analyze_sentiment([speech.speech for speech in transcript_report.dialogue])
        print_transcript_report(transcript, transcript_report, general_sentiment, sentiments)
        dialogues.append((transcript, transcript_report.dialogue, sentiments))
    return dialogues


def print_transcript_report(transcript_file: str, report: TranscriptReport, general_sentiment: Sentiment,
                            sentiments: List[Sentiment],
                            dest_processed_file: Optional[str] = None) -> None:
    path = os.path.abspath(dest_processed_file if dest_processed_file else os.path.dirname(transcript_file))
    base_name, ext = os.path.splitext(os.path.basename(transcript_file))
    dest_file = os.path.join(path, f"{base_name}.txt")
    with open(dest_file, "w") as dest:
        dest.write(f"Job:\t\t{report.job}\nRecording:\t{report.recording}\nTranscript:\t{base_name}{ext}\n"
                   f"Speakers:\t{sorted(report.speakers)}\n")
        dest.write(f"Full Text:\n{report.full_text}\n")
        # noinspection PyProtectedMember
        dest.write(f"Sentiment: {general_sentiment.general}"
                   f" ({', '.join(f'{k[:3]}={v:0.3f}' for (k, v) in general_sentiment._asdict().items() if k != 'general')})\n")
        dest.write("\nDialogue:\n")
        for ((_, _, speaker, speech), sentiment) in zip(report.dialogue, sentiments):
            pred_sentiment: str = sentiment.general
            if sentiment.general == "POSITIVE":
                sentiment_strength = sentiment.positive
            elif sentiment.general == "NEGATIVE":
                sentiment_strength = sentiment.negative
            elif sentiment.general == "NEUTRAL":
                sentiment_strength = sentiment.neutral
            else:
                sentiment_strength = sentiment.mixed
            dest.write(f"{speaker} ({pred_sentiment[:3].lower()}: {sentiment_strength:0.3f}): {speech}\n")


def dialogues_to_df(dialogues: List[Tuple[str, List[Speech], List[Sentiment]]]):
    """
    :param dialogues: List of transcribed dialogues with analysis. Each is (<transcript file name>, <dialogue>, <dialogue sentiment>)
    :return: None
    """
    data = {
        'transcript': [],
        'recording': [],
        'index': [],
        'speaker': [],
        'pred_sent': [],
        'speech': [],
        'positive': [],
        'negative': [],
        'mixed': [],
        'neutral': []
    }
    dialogue: List[Speech]
    sentiments: List[Sentiment]
    for (transcript, dialogue, sentiments) in dialogues:
        for i, (speech, sentiment) in enumerate(zip(dialogue, sentiments)):
            data['transcript'].append(os.path.basename(transcript))
            data['recording'].append(speech.recording)
            data['index'].append(i)
            data['speaker'].append(speech.speaker)
            data['pred_sent'].append(sentiment.general)
            data['speech'].append(speech.speech)
            data['positive'].append(sentiment.positive)
            data['negative'].append(sentiment.negative)
            data['mixed'].append(sentiment.mixed)
            data['neutral'].append(sentiment.neutral)
    df = pd.DataFrame(data)
    return df


def export_df(df: pd.DataFrame, export_path: str,
              export_format: str = "pickle",
              hdf_key: Optional[str] = None
              ) -> None:
    """
    Exports the dataframe in a variety of formats. Dataframe is expected to have columns `transcript` and `index`

    :param df: The dataframe to be exported
    :param export_path: Where to write the exported dataframe
    :param export_format: One of "html", "csv", "json", "parquet", "pickle", "hdf"
    :param hdf_key: if hdf format is used, key to store the df under in the HDF5 file
    """
    _, ext = os.path.splitext(export_path)z
    use_ext = '.' + export_format if len(ext) == 0 else ''
    if export_format == "html":
        color_dict = {
            'POSITIVE': 'limegreen',
            'NEGATIVE': 'red',
            'NEUTRAL': 'lightgrey',
            'MIXED': 'yellow'
        }
        spk_dict = {
            'ch_0': '#F0F8FF',
            'spk_0': '#F0F8FF',
            'ch_1': '#FFF8DC',
            'spk_1': '#FFF8DC'
        }
        df.set_index(
            ['transcript', 'recording', 'speaker', 'index']
        ).to_html(export_path + use_ext, encoding='utf-8',
                  formatters={
                      'pred_sent': lambda sent: f'<span style="background-color:{color_dict[sent]}">{sent}</span>',
                      'speaker': lambda speaker: f'<span style="background-color:{spk_dict[speaker]}">{speaker}</span>'
                  }, escape=False)
    elif export_format == "csv":
        df.to_csv(export_path + use_ext)
    elif export_format == "json":
        df.to_json(export_path + use_ext)
    elif export_format == "parquet":
        df.to_parquet(export_path + use_ext)
    elif export_format == "pickle":
        df.to_pickle(export_path + use_ext)
    elif export_format == "excel":
        if len(use_ext) > 0:
            use_ext = '.xlsx'
        df.to_excel(export_path + use_ext, index=False)
    elif export_format == "hdf":
        assert hdf_key is not None, "Parameter hdf_key must be informed if export format is hdf."
        df.to_hdf(export_path + use_ext, hdf_key)
    else:
        raise ValueError(f"Unknown export format: {export_format}")


if __name__ == "__main__":
    args = parser()
    if args.operation == "transcribe":
        transcribe_recordings(args)
    elif args.operation == "download":
        download_transcripts(args.source_bucket, args.source_path, args.local_path)
    elif args.operation == "process":
        dialogues = process_transcripts(args)
        df = dialogues_to_df(dialogues)
        export_df(df, args.local_df_path, "html")
        export_df(df, args.local_df_path, "pickle")
        export_df(df, args.local_df_path, "excel")
