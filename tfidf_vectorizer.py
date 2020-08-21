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
    X = df.text.apply(lambda tw: clean_tweets(tw, tokenize=False))
    vectorizer.fit(X)
    joblib.dump(vectorizer, f'./models/{model_name}')
    return vectorizer

def validate(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")

if __name__ == '__main__':
    from_date = input('please enter from date, format YYYY-MM-DD')
    to_date = input('please enter to date, format YYYY-MM-DD')
    validate(from_date)
    validate(to_date)
    try:
        df = pd.from_csv(f'./data/tweets_{from_date}_{to_date}.csv')
    except:
        raise FileNotFoundError("Please make sure file exists in data folder!\
            If file is not found, run clean_tweets_data.py to save a file for specified dates.")

    print("Creating Word2Vec Model...")
    model_name = input('please enter model name to save')
    tfidf_vectorizer(df, model_name)

    
    