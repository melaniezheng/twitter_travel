#!/Users/melaniezheng/opt/anaconda3/bin/python

import os
import tweepy 
import pandas as pd
import sqlite3
import time
from datetime import datetime
from sqlite3 import Error
from sqlalchemy import create_engine

consumer_key = os.environ.get('CONSUMER_KEY')
consumer_secret = os.environ.get('CONSUMER_SECRET')
access_token = os.environ.get('ACCESS_TOKEN')
access_token_secret = os.environ.get('ACCESS_TOKEN_SECRET')

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=False)#,parser=tweepy.parsers.JSONParser())

def search_tweets(search_words = "travel OR TRAVEL OR Travel", items = 250):

    # minus retweets
    new_search = search_words + " -filter:retweets"

    # Collect tweets
    try:
        tweets = tweepy.Cursor(api.search,q=new_search,#lang="en",
                            since = '2020-07-01',
                            ).items(items)
    except Exception as e:
        print(f'Failed to retrieve tweets. {e}')

    idstr=[]
    datet=[]
    usern=[]
    verif=[]
    follow_cnt=[]
    friend_cnt=[]
    user_loc=[]
    coordi=[]
    plc=[]
    cntry=[]
    plc_coor=[]
    txt=[]
    num_ret=[]
    num_likes=[]

    for tweet in tweets:
        idstr.append(tweet.id_str)
        datet.append(tweet.created_at)
        usern.append(tweet.user.screen_name)
        verif.append(tweet.user.verified)
        follow_cnt.append(tweet.user.followers_count)
        friend_cnt.append(tweet.user.friends_count)
        user_loc.append(tweet.user.location)
        txt.append(tweet.text)
        num_ret.append(tweet.retweet_count)
        num_likes.append(tweet.favorite_count)

        try:
            coordi.append(tweet.coordinates.coordinates)
        except:
            coordi.append(0)

        try:
            plc.append(tweet.place.fullname)
        except:
            plc.append(0)

        try:
            cntry.append(tweet.place.country)
        except:
            cntry.append(0)

        try:
            plc_coor.append(tweet.place.bounding_box.coordinates)
        except:
            plc_coor.append(0)

    search_result = list(zip(idstr, datet, usern, verif, follow_cnt, friend_cnt, user_loc, coordi, plc, cntry, plc_coor, txt, num_ret, num_likes))

    tweet_text = pd.DataFrame(data=search_result, 
                        columns=['id_str','datetime','username', 'verified', 'followers_count','friends_count',
                                 'user_location', 'coordinates', 'place', 'country', 'place_coordinates',
                                 'text', 'num_retweets', 'num_likes'])

    # from datetime import datetime
    # import time
    # t = datetime.fromtimestamp(time.time())
    # s = t.strftime("%Y-%m-%d_%H:%M")

    # tweet_text.to_csv('traveltweet-tmp.csv', index=False)

    return tweet_text


def insert_data(df,  db_file, tablename='TWEETS'):
    # prevent coordinate dtype error inserting into sqlite
    df.coordinates = df.coordinates.apply(lambda x : str(x))
    df.place_coordinates = df.place_coordinates.apply(lambda x : str(x))
    
    engine = create_engine(f'sqlite:///{db_file}', echo=False)
    df.to_sql(tablename, con=engine, if_exists='append', index=False)

if __name__ == "__main__":
    t = datetime.fromtimestamp(time.time())
    s = t.strftime("%Y-%m-%d_%H:%M")
    print(f'{s} - Starting Main...')
    insert_data(df = search_tweets(), db_file = 'traveltweets.db')
    t = datetime.fromtimestamp(time.time())
    s = t.strftime("%Y-%m-%d_%H:%M")
    print(f'{s} - Complete!')
