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
    train_df = train_df[~train_df.clean_text.isna()]
    vectors = vectorizer.transform(train_df.clean_text)
    feature_names = vectorizer.get_feature_names()

    # convert tfidf vectors to dataframe
    tfidf = pd.DataFrame(vectors.toarray(), columns=feature_names)
    w2v_vocabs = list(model.wv.vocab.keys())

    # for each tweet we define feature vectors as:
    # the mean of w2v vector * tf-idf value for each word in the tweet.
    wv = model.wv
    wv_num_features = wv.vectors[0].shape[0]
    w2v_tfidf_mean = []
    for i, tweet in enumerate(df.clean_text):
        n = 0
        v = np.zeros(wv_num_features,)
        for word in tweet.split():
            if word in w2v_vocabs and word in feature_names:
                n += 1
                w2v = wv[word]
                ti = tfidf.iloc[i][word]
                v += w2v*ti
        try:
            w2v_tfidf_mean.append(v/n)
        except ZeroDivisionError:
            w2v_tfidf_mean.append(0)
    # generate feature dataframe
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


def generate_features(train_df, stocks_df, w2vmodel="./models/w2v_model", tfVectorizer='./models/tfVectorizer',  interval = '15'):
    features = generate_word_feature_vectors(w2vmodel, tfVectorizer, train_df)
    X = pd.merge(features.reset_index(), stocks_df, left_on ='datetime', right_on ='date').drop(columns=['date'])
    y = X.pop('target')
    dt = X.pop('datetime')

    return X, y, dt


def train(X, y):
    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(random_state=1234, max_depth = 5)
    rfr.fit(X, y)
    joblib.dump(rfr, f'./models/random_forest.m')
    print('Model saved!')

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
    tweets_file = input('Enter tweets file name: ')
    stocks_file = input('Enter stocks file name: ')

    df = validate_file('tweets', tweets_file)
    stocks_df = validate_file('stocks', stocks_file)

    print("Generating Features...")
    X, y, dt = generate_features(df, stocks_df)
    print('Trainig Model...')
    train(X,y)
    print('Done!')
    
    