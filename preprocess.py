import libs.preprocessing


def main():
    import pandas as pd

    train_data = pd.read_csv(
        './assets/Corona_NLP_train.csv', encoding='ISO-8859-1')
    test_data = pd.read_csv(
        './assets/Corona_NLP_test.csv', encoding='ISO-8859-1')

    train_data['OriginalTweet'] = train_data['OriginalTweet'].astype(str)
    train_data['Sentiment'] = train_data['Sentiment'].astype(str)

    test_data['OriginalTweet'] = test_data['OriginalTweet'].astype(str)
    test_data['Sentiment'] = test_data['Sentiment'].astype(str)

    train_data['text'] = train_data['OriginalTweet']
    train_data['text'] = train_data['text'].astype(str)

    test_data['text'] = test_data['OriginalTweet']
    test_data['text'] = test_data['text'].astype(str)

    train_data['text'] = train_data['text'].apply(
        libs.preprocessing.preprocess_text)
    test_data['text'] = test_data['text'].apply(
        libs.preprocessing.preprocess_text)

    train_data.to_csv('./assets/Corona_NLP_train_clean.csv', index=False)
    test_data.to_csv('./assets/Corona_NLP_test_clean.csv', index=False)


if __name__ == '__main__':
    main()
