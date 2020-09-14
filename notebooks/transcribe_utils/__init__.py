import json
import os
from collections import Counter, namedtuple
from os.path import basename
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import boto3
from implicits import implicits

Speech = namedtuple('Speech', ['job', 'recording', 'speaker', 'speech'])

TranscriptReport = namedtuple('TranscriptReport', ['job', 'recording', 'speakers', 'full_text', 'dialogue'])


@implicits('bucket', 'path')
def get_recording_files(bucket: str, path: str) -> List[str]:
    """Retrieves the file names for all the recordings in the specified path"""
    s3 = boto3.client('s3')
    files = s3.list_objects_v2(Bucket=bucket, Prefix=path)
    fnames = [f['Key'] for f in files['Contents']][1:]  # The first value is the folder name
    return [basename(fname) for fname in fnames]


@implicits('transcribe', 'bucket', 'path')
def transcribe_recording(
    recording: str,
    job_name: str,
    bucket: str,
    path: str,
    transcribe: Any,
    **settings
) -> Dict[str, str]:
    print(f"Starting job {job_name} for s3://{bucket}/{path}/{recording}")
    settings = {
        **{
            "ChannelIdentification": True,
            "ShowAlternatives": True,
            "MaxAlternatives": 2,
            "VocabularyName": "dpv-keywords"
        },
        **settings
    }
    return transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        LanguageCode="de-DE",
        MediaFormat="wav",
        OutputBucketName=bucket,
        Settings=settings,
        Media={
            "MediaFileUri": f"s3://{bucket}/{path}/{recording}"
        }
    )


def report_job_results(jobs: Dict[str, Tuple[str, Dict[str, Dict[str, Any]]]]):
    for job_name, (final_status, result) in jobs.items():
        result = result['TranscriptionJob']
        print(f"Job {job_name} finished with status {final_status}\n"
              f"\tMedia: {result['Media']['MediaFileUri']}"
              )
        try:
            print(f"\tTranscript: {result['Transcript']['TranscriptFileUri']}\n")
        except KeyError:
            print(f"\tFailure reason: {result['FailureReason']}")


@implicits('transcribe')
def wait_for_jobs(jobs: List[Dict[str, Any]], transcribe: Any, sleep_time=10) -> Dict[str, Tuple[str, Dict[str, str]]]:
    job_statuses = {}
    print('Waiting')
    while len(job_statuses) == 0 or any(j not in ['COMPLETED', 'FAILED'] for (j, _) in job_statuses.values()):
        sleep(sleep_time)
        for job in jobs:
            job_name = job['TranscriptionJob']['TranscriptionJobName']
            status = None
            if job_statuses.get(job_name, ('dummy', {}))[0] not in ('COMPLETED', 'FAILED'):
                result = transcribe.get_transcription_job(TranscriptionJobName=job_name)
                status = result['TranscriptionJob']['TranscriptionJobStatus']
                job_statuses[job_name] = (status, result)
            else:
                continue
            if status and status == 'IN_PROGRESS':
                print('At least one in progress, wait more')
                break
    report_job_results(job_statuses)
    return job_statuses


def _get_path_from_uri(transcript_uri):
    path, file = os.path.split(transcript_uri)
    bucket = path[8:].split('/')[1]
    path = '/'.join(path[8:].split('/')[2:])
    if len(path):
        path += "/"
    return bucket, path, file


def move_transcripts(jobs: Dict[str, Tuple[str, Dict[str, Any]]], dest_bucket: str, dest_path: str) -> None:
    s3_res = boto3.resource('s3')
    s3_cli = boto3.client('s3')
    dest = s3_res.Bucket(dest_bucket)
    for job_name, (_, job_info) in jobs.items():
        job_info = job_info['TranscriptionJob']
        transcript_uri = job_info['Transcript']['TranscriptFileUri']
        bucket, path, file = _get_path_from_uri(transcript_uri)
        if len(dest_path) and dest_path[-1] != '/':
            dest_path += '/'
        dest.copy(
            CopySource={
                'Bucket': bucket,
                'Key': f"{path}{file}"
            },
            Key=f"{dest_path}{file}"
        )
        s3_cli.delete_object(Bucket=bucket, Key=f"{path}{file}")


def download_transcripts(source_bucket: str, source_path: str, dest_path: str) -> None:
    bucket_name = source_bucket
    path = source_path
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    if not os.path.exists(dest_path):
        print(f"{dest_path} does not exist, creating...")
        os.makedirs(dest_path)
    for transcript in bucket.objects.filter(Prefix=path):
        _, fname = os.path.split(transcript.key)
        if len(fname) == 0:
            continue
        print(f"Downloading {transcript.key} into {dest_path}/{fname}")
        bucket.download_file(transcript.key, f"{dest_path}/{fname}")


@implicits('transcribe')
def delete_job_and_result(job_name: str, transcribe: Any) -> bool:
    job = transcribe.get_transcription_job(TranscriptionJobName=job_name)
    transcript_uri = job['TranscriptionJob']['Transcript'].get('TranscriptFileUri', None)
    if job['TranscriptionJob']['TranscriptionJobStatus'] in {'FAILED', 'COMPLETED'}:
        print(f"Deleting job {job_name}")
        transcribe.delete_transcription_job(TranscriptionJobName=job_name)
        if transcript_uri:
            s3 = boto3.client('s3')
            bucket, file, path = _get_path_from_uri(transcript_uri)
            print(f"Deleting {path}/{file} from {bucket}")
            s3.delete_object(Bucket=bucket, Key=f"{path}{file}")
        return True
    else:
        print(f"Skiping job {job_name}, still {job['TranscriptionJob']['TranscriptionJobStatus']}")
        return False


def delete_all_jobs(prefix: Optional[str] = None, max_results: int = 100):
    transcribe = boto3.client('transcribe')
    jobs = (
        job["TranscriptionJobName"]
        for job in transcribe.list_transcription_jobs(
        MaxResults=max_results,
        JobNameContains=prefix
    )["TranscriptionJobSummaries"]
    )
    for job in jobs:
        delete_job_and_result(job)


@implicits('transcribe')
def report_transcript(
    transcript_file: str,
    transcribe: Any,
    dest_processed_file: Optional[str] = None
) -> TranscriptReport:
    with open(transcript_file, "r") as tf:
        transcript = json.load(tf)
    assert transcript['status'] == "COMPLETED", f"Transcript {transcript_file} is not from a completed job."
    split_channels = 'channel_labels' in transcript['results']

    job = transcript['jobName']
    recording = os.path.basename(
        transcribe.get_transcription_job(TranscriptionJobName=job)['TranscriptionJob']['Media']['MediaFileUri']
    )
    full_text = " ".join(t['transcript'] for t in transcript['results']['transcripts'])
    segments = transcript['results']['segments']
    speakers, speeches = _extract_split_channels(
        job, recording,
        transcript['results']['channel_labels']['channels'],
        segments
    ) if split_channels else _extract_speaker_identification(
        job, recording,
        transcript['results']['speaker_labels']['segments'],
        segments
    )
    return TranscriptReport(job, recording, speakers, full_text, speeches)


def _extract_speaker_identification(job, recording, speaker_segments, segments):
    # Speech processing loop initialization
    speeches = []
    utterances = []
    speaker = speaker_segments[0]['speaker_label']
    speakers = {speaker}
    speech_processor = zip(speaker_segments, segments)
    cur_speaker, cur_segment = next(speech_processor)
    try:
        while True:  # Infinite loop will be broken when speech_processor is exhausted
            # While we're with the same speaker, collect all utterances
            while cur_speaker['speaker_label'] == speaker:
                utterances.append(cur_segment['alternatives'][0]['transcript'])
                cur_speaker, cur_segment = next(speech_processor)
            # Save all contiguous utterances of the same speaker as one speech.
            speeches.append(Speech(job, recording, speaker, " ".join(utterances)))
            # Prepare to start the next speaker's speech
            speaker = cur_speaker['speaker_label']
            speakers.add(speaker)
            utterances = []
    except StopIteration:
        # Save the last speech
        speeches.append(Speech(job, recording, speaker, " ".join(utterances)))
    return speakers, speeches


def _extract_split_channels(job, recording, channel_segments, segments):
    # Speech processing loop initialization
    speeches = []
    speakers = {channel['channel_label'] for channel in channel_segments}
    for segment in segments:
        segment_start = float(segment['start_time'])
        segment_end = float(segment['end_time'])
        segment_transcript = segment['alternatives'][0]['transcript']
        segment_channel_labels = [
            channel['channel_label']
            for channel in channel_segments
            for channel_item in channel['items']
            if (channel_item['type'] == "pronunciation" and
                float(channel_item['start_time']) >= segment_start and
                float(channel_item['end_time']) <= segment_end
                )
        ]
        top_channel_label = Counter(segment_channel_labels).most_common(1)[0][0]
        speeches.append(Speech(job, recording, top_channel_label, segment_transcript))
    return speakers, speeches
