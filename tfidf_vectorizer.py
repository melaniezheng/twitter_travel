import os
import pandas as pd
import numpy as np
import joblib
import datetime
from clean_tweets_data import clean_tweets

from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorizer(df, model_name, num_features = 200):
    # train tfidf vectorizer 
    vectorizer = TfidfVectorizer()
    df = df[~df.clean_text.isna()]
    vectorizer.fit(df.clean_text)
    joblib.dump(vectorizer, f'./models/{model_name}')
    return vectorizer

def validate_file(type, filename):
    if type == 'tweets':
        try:
            df = pd.read_csv(f'./data/{filename}', \
                converters={"clean_text": lambda x: x.strip("[]").split(", ")})
        except FileNotFoundError:
            raise FileNotFoundError(f"{filename} Not Found!")
    elif type == 'stocks':
        try:
            df = pd.read_csv(f'./data/{filename}', parse_dates=['date'])
        except FileNotFoundError:
            raise FileNotFoundError(f"{filename} Not Found!")
    else:
        raise ValueError(f'Incorrect type. Choose from ["tweets", "stocks"]')
    return df

if __name__ == '__main__':
    tweet_file = input('please enter tweet filename: ')
    df = validate_file('tweet', tweet_file)
    print("Creating Word2Vec Model...")
    model_name = input('please enter model name to save')
    tfidf_vectorizer(df, model_name)

