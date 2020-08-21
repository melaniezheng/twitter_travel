import os
import datetime
import pandas as pd
import numpy as np
import joblib
from gensim.models import word2vec
import pickle
from clean_tweets_data import clean_tweets


def generate_word_feature_vectors(w2vmodel, tfVectorizer, train_df, interval = '15'):
    # generate feature vectors from tweets.
    # size depends on w2v model feature size.
    # currently we have 200 feature vectors per tweet + 3 additional hand crafted features 
    model = word2vec.Word2Vec.load(w2vmodel, mmap='r')
    vectorizer = joblib.load(tfVectorizer)
    clean_tweets = train_df.text.apply(lambda t: clean_tweets(t, tokenize = False))
    vectors = vectorizer.transform(clean_tweets)
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
    for i, tweet in enumerate(train_df.clean_tweets):
        n = len(tweet)
        v = np.zeros(200,)
        for word in tweet:
            if word in w2v_vocabs and word in feature_names:
                w2v = wv[word]
                ti = tfidf.iloc[i][word]
                v += w2v*ti
        w2v_tfidf_mean.append(v/n)
    X = pd.DataFrame(w2v_tfidf_mean, columns=['feat_'+str(i) for i in range(wv_num_features)])
    
    
    # add additional features i.e. high_followers, high_num_tweets (1 for high 0 for low), 
    high_followers = train_df.followers_count.describe()['75%']
    train_df['high_followers'] = train_df['followers_count'].apply(lambda x: 1 if x >= high_followers else 0)
    X = pd.concat([train_df[['verified','high_followers']], X], axis = 1)

    X['datetime'] = pd.to_datetime(train_df['datetime'])
    X['datetime'] = X['datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour,interval*(dt.minute // interval)))
    
    # aggregate to specified time interval i.e. 15 minute interval
    features = X.groupby('datetime').mean()
    features['num_tweets'] = X.groupby('datetime').size().tolist()
    high_num_tweets = features.num_tweets.describe()['75%']
    features['high_num_tweets'] = features['num_tweets'].apply(lambda x: 1 if x >= high_num_tweets else 0)
    features.drop(columns='num_tweets', inplace=True)

    # save parameters for preprocessing
    params = {}
    params['high_followers'] = high_followers
    params['high_num_tweets'] = high_num_tweets
    pickle.dump(params, open( "./models/params.p", "wb" ))

    return features


def generate_features(train_df, stocks_df, w2vmodel="./model/w2v_model", tfVectorizer='./model/tfVectorizer',  interval = '15'):
    features = generate_word_feature_vectors(w2vmodel, tfVectorizer, train_df)
    X = pd.merge(features.reset_index(), stocks_df, left_on ='datetime', right_on ='date').drop(columns=['date'])
    y = X.pop('target')
    X = X.pop('datetime')

    return X, y


def train(X, y):
    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(random_state=1234, max_depth = 5)
    rfr.fit(X, y)
    joblib.dump(rfr, f'./models/random_forest.m')
    print('Model saved!')


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
        df = pd.read_csv(f'./data/tweets_{from_date}_{to_date}.csv')
        stocks_df = pd.read_csv(f'./data/JETS_{from_date}_{to_date}.csv',parse_dates=['date'])
    except:
        raise FileNotFoundError("Please make sure file exists in data folder!\
            If file is not found, run clean_tweets_data.py to save a file for specified dates.")
    print("Generating Features...")
    X, y = generate_features(df, stocks_df)
    print('Trainig Model...')
    train(X,y)
    print('Done!')
    
    