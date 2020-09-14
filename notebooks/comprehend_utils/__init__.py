from collections import Counter, namedtuple
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Tuple, Union

from implicits import implicits

Sentiment = namedtuple('Sentiment', ['general', 'positive', 'negative', 'neutral', 'mixed'])


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


@implicits('comprehend')
def analyze_sentiment(
    text: Optional[Union[str, List[str]]], comprehend: Any, language_code: str = 'de'
) -> Union[Sentiment, List[Sentiment]]:
    def merge_responses(response_list):
        result_list = []
        error_list = []
        offset = 0
        for response in response_list:
            results = response['ResultList']
            errors = response['ErrorList']
            number_docs = len(results) + len(errors)
            offset_results = [
                {
                    'Index': result['Index'] + offset,
                    'Sentiment': result['Sentiment'],
                    'SentimentScore': result['SentimentScore']
                }
                for result in results
            ]
            offset_errors = [
                {
                    'Index': error['Index'] + offset,
                    'ErrorCode': error['ErrorCode'],
                    'ErrorMessage': error['ErrorMessage']
                }
                for error in errors
            ]
            result_list += offset_results
            error_list += offset_errors
            offset += number_docs
        return {
            'ResultList': result_list,
            'ErrorList': error_list
        }


    string_input = isinstance(text, str)
    text_lists = grouper([text] if string_input else text, 25)
    response_list = []
    for text_list in text_lists:
        response_list.append(comprehend.batch_detect_sentiment(
            TextList=[text for text in text_list if text is not None],
            LanguageCode=language_code
        ))
    response = merge_responses(response_list)
    sentiments = {
        result['Index']: (
            result['Sentiment'],
            result['SentimentScore']['Positive'],
            result['SentimentScore']['Negative'],
            result['SentimentScore']['Neutral'],
            result['SentimentScore']['Mixed']
        )
        for result in response['ResultList']
    }
    errors = {
        result['Index']: (
            result['ErrorCode'],
            result['ErrorMessage']
        )
        for result in response['ErrorList']
    }
    sentiments = {**sentiments, **errors}
    results = [
        Sentiment(sentiment[0], sentiment[1], sentiment[2], sentiment[3], sentiment[4])
        if len(sentiment) == 5 else Sentiment('ERROR', sentiment[0], sentiment[1], None, None)
        for _, sentiment in sorted(sentiments.items())
    ]
    if len(results) > 1:
        return results
    else:
        return results[0]
