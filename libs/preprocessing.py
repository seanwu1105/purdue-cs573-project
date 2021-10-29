import re

import nltk.corpus

STOPWORDS = set(nltk.corpus.stopwords.words('english'))


def classify_sentiment(text: str):
    if text == 'Extremely Positive':
        return '2'
    elif text == 'Extremely Negative':
        return '0'
    elif text == 'Negative':
        return '0'
    elif text == 'Positive':
        return '2'
    return '1'


def preprocess_text(text: str):
    text = remove_urls(text)
    text = remove_html(text)
    text = lower(text)
    text = remove_numbers(text)
    text = remove_punctuations(text)
    text = remove_stopwords(text)
    text = remove_mention(text)
    text = remove_hashtags(text)
    text = remove_space(text)
    return text


def remove_urls(text: str):
    url_remove = re.compile(r'https?://\S+|www\.\S+')
    return url_remove.sub(r'', text)


def remove_html(text: str):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def lower(text: str):
    low_text = text.lower()
    return low_text


def remove_numbers(text: str):
    remove = re.sub(r'\d+', '', text)
    return remove


def remove_punctuations(text: str):
    punct = re.sub(r'[^\w\s\d]', '', text)
    return punct


def remove_stopwords(text: str):
    '''custom function to remove the stopwords'''
    return ' '.join([word
                     for word in str(text).split()
                     if word not in STOPWORDS])


def remove_mention(text: str):
    text = re.sub(r'@\w+', '', text)
    return text


def remove_hashtags(text: str):
    text = re.sub(r'#\w+', '', text)
    return text


def remove_space(text: str):
    space_remove = re.sub(r'\s+', ' ', text).strip()
    return space_remove
