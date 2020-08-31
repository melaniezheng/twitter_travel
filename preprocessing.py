
import datetime
import pandas as pd
import numpy as np
from textblob import TextBlob
import joblib
from sklearn.preprocessing import MinMaxScaler

def get_sentiment(text):
    tb = TextBlob(text)
    return TextBlob(text).sentiment

def training_preprocessor(tweets_df, stocks_df, train_size, interval = 15, scale = True):
    '''generate features from dataframe
    final features: verified, followers_count, tweet_count, has_locations, last_close, last_vol, last_pct_change
    input: tweet_df, stocks_df, output_filename
    output: x_train, y_train, x_test, y_test, train_datetime, test_datetime'''
    polarities = []
    sentiments = []
    for t in tweets_df.clean_text:
        p,s = get_sentiment(str(t))
        polarities.append(p)
        sentiments.append(s)
        
    tweets_df['polarity'] = polarities
    tweets_df['sentiment'] = sentiments

    X = tweets_df.copy()
    X['datetime'] = pd.to_datetime(X['datetime'], format="%Y-%m-%d %H:%M:%S", errors = 'coerce') # convert to datetime format
    X = X.dropna(subset = ['datetime']) # drop missing dates
    X['datetime'] = X['datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour,interval*(dt.minute // interval)))

    # add new features: average of verified tweets, average follower counts, average percent user with location tag
    features = X.groupby('datetime').mean()[['polarity','sentiment','verified', 'followers_count']].reset_index()
    features['tweet_count'] = X.groupby('datetime').size().tolist()
    features['has_locations'] = X.groupby('datetime')['user_location'].count().tolist()
    features['has_locations'] = features['has_locations'] / features['tweet_count']

    # merge stocks data
    features = pd.merge(features, stocks_df, left_on='datetime', right_on = 'date').drop(columns=['date'])
    features.to_csv(f'./data/tmp/features_train_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv', index = False)
    # features should have datetime, verified, followers_count, tweet_count, has_locations, \
    # last_close, last_vol, last_pct_change, target
    train = features[:176]
    test = features[176:]
    x_train = train.copy()
    x_test = test.copy()
    dt_train = x_train.pop('datetime')
    dt_test = x_test.pop('datetime')
    y_train = x_train.pop('target')
    y_test = x_test.pop('target')

    # min_max scale
    if scale:
        sc = MinMaxScaler(feature_range=(0,1))
        x_train = sc.fit_transform(x_train) # except Datetime column
        joblib.dump(sc, './models/min_max_scaler')
        x_test = sc.transform(x_test)

    return x_train, y_train, x_test, y_test, dt_train, dt_test

def prediction_preprocessor(tweets_df, stocks_df, interval = 15, scale = True):
    '''generate features from dataframe
    final features: verified, followers_count, tweet_count, has_locations, last_close, last_vol, last_pct_change
    input: tweet_df, stocks_df, output_filename
    output: features(X)'''
    polarities = []
    sentiments = []
    for t in tweets_df.clean_text:
        p,s = get_sentiment(str(t))
        polarities.append(p)
        sentiments.append(s)
        
    tweets_df['polarity'] = polarities
    tweets_df['sentiment'] = sentiments
    X = tweets_df.copy()
    X['datetime'] = pd.to_datetime(X['datetime'], format="%Y-%m-%d %H:%M:%S", errors = 'coerce') # convert to datetime format
    X = X.dropna(subset = ['datetime']) # drop missing dates
    X['datetime'] = X['datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour,interval*(dt.minute // interval)))

    # add new features: average of verified tweets, average follower counts, average percent user with location tag
    features = X.groupby('datetime').mean()[['polarity','sentiment','verified', 'followers_count']].reset_index()
    features['tweet_count'] = X.groupby('datetime').size().tolist()
    features['has_locations'] = X.groupby('datetime')['user_location'].count().tolist()
    features['has_locations'] = features['has_locations'] / features['tweet_count']

    # merge stocks data
    features = pd.merge(features, stocks_df, left_on='datetime', right_on = 'date').drop(columns=['date'])
    features.to_csv(f'./data/tmp/features_pred_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv', index = False)
    # features should have datetime, verified, followers_count, tweet_count, has_locations, \
    # last_close, last_vol, last_pct_change
    X = train.copy
    dt = train.pop('datetime')

    if scale:
        sc = joblib.load('./models/min_max_scaler')
        X = sc.transform(X)
    return X, dt


def validate_file(type, filename):
    if type == 'tweets':
        try:
            df = pd.read_csv(f'./data/{filename}')
        except FileNotFoundError:
            raise FileNotFoundError(f"{filename} Not Found!")
    elif type == 'stocks':
        try:
            df = pd.read_csv(f'./data/{filename}', parse_dates = ['date'])
        except FileNotFoundError:
            raise FileNotFoundError(f"{filename} Not Found!")
    else:
        raise ValueError(f'Incorrect type. Choose from ["tweets", "stocks"]')
    return df

def run_preprocessor(type, tweets_file, stocks_file, train_size=176, interval = 15):
    '''input:
    type: string, "training" or "predicting",
    tweets_file: filepath to tweets file
    stocks_file: filepath to stocks file'''
    if type == 'training':
        tweets_df = validate_file('tweets', tweets_file)
        stocks_df = validate_file('stocks', stocks_file)
        X_train, y_train, X_test, y_test, train_dt, test_dt = training_preprocessor(tweets_df, stocks_df, train_size, interval)
        return X_train, y_train, X_test, y_test, train_dt, test_dt
    elif type == 'predicting':
        tweets_df = validate_file('tweets', tweets_file)
        stocks_df = validate_file('stocks', stocks_file)
        X = prediction_preprocessor(tweets_df, stocks_df, interval)
        return X
    else:
        raise ValueError('Incorrect type provided. Please enter "training" or "predicting".')
    
    