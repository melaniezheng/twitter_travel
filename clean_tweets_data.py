import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine

# for string processing
import re
import copy
import string
from langdetect import detect
from googletrans import Translator
from langdetect import detect
import json

import preprocessor as p # Currently supports cleaning, tokenizing and parsing of tweets: https://pypi.org/project/tweet-preprocessor/
import demoji # Reference: https://pypi.org/project/demoji/
demoji.download_codes()
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def translate(tweet):
    '''maybe one-day we can afford unlimited google translate api access'''
    trans = Translator()
    to_eng = trans.translate(tweet)
    return to_eng.text

def clean_tweets(tweet, translate = False, tokenize = False):
    '''process tweets:
    - tweet processor to get rid of url links
    - remove punctuations
    - translate to english
    - to lowercase
    - remove common stopwords
    - stem instead of lemmitize (for faster process, although draw back is the stemmed words sometime do not look like a real word)
    - process emojis i.g. :) --> smily face'''
    tweet = str(tweet) # sometimes not a string type i.e. float
    orig_tweet = copy.deepcopy(tweet)
    # remove urls,  mentions, and hashtags
    try:
        tweet = p.clean(tweet) 
    except Exception as e:
        print(e)
    # 1. remove non-letter or space.
    tweet = re.sub('[^[a-z|A-Z|\s]*', '', tweet)
    if translate:
        try:
            # translate to English
            tweet = translate(tweet)
        except json.decoder.JSONDecodeError:
            print(tweet)
    
    stop = stopwords.words('english') # common stop words
    stop_words = set(stop)
    # 2. convert to lower case and tokenize
    tweet = tweet.lower()
    tweet_tokens = word_tokenize(tweet)
    # 3. remove stopwords, Stemming
    ret = []
    for word in tweet_tokens:
        if not word in stop_words:
            stemmer = PorterStemmer()
            word = stemmer.stem(word)
            ret.append(word)
    # 4. append emojis
    ret.extend(list(demoji.findall(orig_tweet).values())) 
    if not tokenize:
        ret = ' '.join(ret)
    return ret

def get_tweets_data(from_date, to_date, db_file = 'traveltweets.db'):
    '''get tweets from sqllite database. 
    FIELDS: 'id_str','datetime','username', 'verified', 'followers_count','friends_count',
    'user_location', 'coordinates', 'place', 'country', 'place_coordinates',
    'text', 'num_retweets', 'num_likes'
    '''
    engine = create_engine(f'sqlite:///{db_file}', echo=False) 
    df = pd.read_sql_query(sql='SELECT * FROM TWEETS;', con=engine)
    print(f'original dataframe shape: {df.shape}')

    df = df.sort_values(by = ['datetime', 'username'], ascending = [False, True])
     # keep only most recent tweets if multiple tweets exist
    df = df.drop_duplicates(subset='text')
    # shuffle
    df = df.sample(frac = 1, random_state = 1234) 
    df.dropna(subset=['text'], how='any', axis=0, inplace=True)
    print(f'clean dataframe shape; {df.shape}')

    assert 0 == sum(df.text.isnull())

    lang_detect = []
    for tw in df.text:
        try:
            lang_detect.append(detect(tw))
        except:
            lang_detect.append('NotFound')
            
    df['lang'] = lang_detect
    df['clean_text'] = df.text.apply(lambda tw: clean_tweets(tw, tokenize=True))
    csv_fn = f'tweets_{from_date}_{to_date}.csv'
    df.to_csv(csv_fn, index=False) # save to csv file
    print(f'{csv_fn} created successfully!')

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
    get_tweets_data(from_date, to_date)


    
    