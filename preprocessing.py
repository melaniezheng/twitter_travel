import os
import datetime
import pandas as pd
import numpy as np
import joblib
from gensim.models import word2vec
import pickle
from clean_tweets_data import clean_tweets


def generate_word_feature_vectors(w2vmodel, tfVectorizer, df, params, interval = 15):
    # generate feature vectors from tweets.
    # size depends on w2v model feature size.
    # currently we have 200 feature vectors per tweet + 3 additional hand crafted features 
    model = word2vec.Word2Vec.load(w2vmodel, mmap='r')
    vectorizer = joblib.load(tfVectorizer)
    tweets = df.clean_tweets.apply(lambda l: ' '.join(l))
    vectors = vectorizer.transform(tweets)
    feature_names = vectorizer.get_feature_names()

    # convert tfidf vectors to dataframe
    tfidf = pd.DataFrame(vectors.toarray(), columns=feature_names)
    w2v_vocabs = list(model.wv.vocab.keys())
    missing_word = []
    # drop tfidf tokens that are not in w2v tokens
    for word in feature_names:
        if word not in w2v_vocabs:
            missing_word.append(word)
    tfidf.drop(columns=missing_word, inplace=True)

    # for each tweet we define feature vectors as the mean of w2v vector * tf-idf value for each word in the tweet.
    wv = model.wv
    wv_num_features = wv.vectors[0].shape[0]
    w2v_tfidf_mean = []
    for i, tweet in enumerate(tweets):
        n = len(tweet.split())
        v = np.zeros(200,)
        for word in tweet.split():
            if word in w2v_vocabs and word in feature_names:
                w2v = wv[word]
                ti = tfidf.iloc[i][word]
                v += w2v*ti
        w2v_tfidf_mean.append(v/n)
    X = pd.DataFrame(w2v_tfidf_mean, columns=['feat_'+str(i) for i in range(wv_num_features)])
    X.to_csv('./data/tmp/feature_vectors.csv', index = False)
    
    
    # add additional features i.e. high_followers, high_num_tweets (1 for high 0 for low), 
    high_followers = params['high_followers'] 
    df['high_followers'] = df['followers_count'].apply(lambda x: 1 if x >= high_followers else 0)
    X = pd.concat([df[['verified','high_followers']], X], axis = 1)


    X['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S", errors = 'coerce')
    X = X.dropna(subset = ['datetime']) # drop missing dates
    X['datetime'] = X['datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour,interval*(dt.minute // interval)))
    
    # aggregate to specified time interval i.e. 15 minute interval
    features = X.groupby('datetime').mean()
    features['num_tweets'] = X.groupby('datetime').size().tolist()
    high_num_tweets = params['high_num_tweets']
    features['high_num_tweets'] = features['num_tweets'].apply(lambda x: 1 if x >= high_num_tweets else 0)
    features.drop(columns='num_tweets', inplace=True)

    return features


def generate_features(df, stocks_df, params, w2vmodel="./models/w2v_model", tfVectorizer='./models/tfVectorizer',  interval = 15):
    features = generate_word_feature_vectors(w2vmodel, tfVectorizer, df, params)
    X = pd.merge(features.reset_index(), stocks_df, left_on ='datetime', right_on ='date').drop(columns=['date'])
    y = X.pop('target')
    dt = X.pop('datetime')
    return X, y, dt

def validate(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")

def run_preprocessor(from_date, to_date):
    validate(from_date)
    validate(to_date)
    try:
        df = pd.read_csv(f'./data/tweets_{from_date}_{to_date}.csv')
        stocks_df = pd.read_csv(f'./data/JETS_{from_date}_{to_date}.csv',parse_dates=['date'])
    except:
        raise FileNotFoundError("Please make sure file exists in data folder! If the file is not found, run clean_tweets_data.py to save a file for specified dates.")
    params = pickle.load(open('./models/params.p','rb'))
    X, y, dt = generate_features(df, stocks_df, params)
    return X, y, dt
    
    