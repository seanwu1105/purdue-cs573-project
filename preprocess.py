import pandas as pd

import libs.preprocessing


def main():

    train_data = pd.read_csv(
        './assets/Corona_NLP_train.csv', encoding='ISO-8859-1')
    test_data = pd.read_csv(
        './assets/Corona_NLP_test.csv', encoding='ISO-8859-1')

    df = pd.concat([train_data, test_data])

    df['OriginalTweet'] = df['OriginalTweet'].astype(str)
    df['Sentiment'] = df['Sentiment'].astype(str)

    df['text'] = df['OriginalTweet']
    df['text'] = df['text'].astype(str)

    df['text'] = df['text'].apply(libs.preprocessing.preprocess_text)
    df['label'] = df['Sentiment'].apply(libs.preprocessing.classify_sentiment)

    df.to_csv('./assets/preprocessed.csv', index=False,
              encoding='ISO-8859-1', line_terminator='\r\n')


if __name__ == '__main__':
    main()
