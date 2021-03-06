#!/Users/melaniezheng/opt/anaconda3/bin/python

'''cronjob schedule to run every 2 minutes:
ex: */2 * * * * . /Users/melaniezheng/.bash_profile; cd /twitter_travel/ && ./get_tweets.py >> log.txt
'''
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

def search_tweets(search_words = "travel OR TRAVEL OR Travel", items = 1000):
    ''' search travel related tweets from twitter api,
    api limit: 180 requests/15 minute, 100 tweets/request'''
    # minus retweets
    new_search = search_words + " -filter:retweets"

    # Collect tweets
    try:
        tweets = tweepy.Cursor(api.search,q=new_search,#lang="en",
                            since = '2020-07-01',
                            ).items(items)


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
        print(f'done fetching data! {tweet_text.shape}')
        return tweet_text

    except Exception as e:
        print(f'Failed to retrieve tweets. {e}')



def insert_data(df,  db_file, tablename='TWEETS'):
    # prevent coordinate dtype error inserting into sqlite
    df.coordinates = df.coordinates.apply(lambda x : str(x))
    df.place_coordinates = df.place_coordinates.apply(lambda x : str(x))
    
    engine = create_engine(f'sqlite:///{db_file}', echo=False)
    df.set_index('id_str', inplace=True)
    
    '''following blocks are performing manual UPSERT
    pandas to_sql does not support UPSERT for SQlite'''
    
    # pull existing tweets in the db
    df_update = pd.read_sql_query(sql='SELECT * FROM AMD;', con=engine, index_col = 'id_str')
    # update existing records
    df_update.update(df)
    
    dup_rows = pd.merge(df, df_update,  left_index = True, right_index = True).index
    new_data = df[~df.index.isin(dup_rows)] # get new records
    
    df_update.to_sql('AMD', con=engine, if_exists='replace', index=True) # update existing records
    new_data.to_sql('AMD', con=engine, if_exists='append', index=True) # insert new records
    
    print(f'{new_data.shape[0]} new data inserted')

if __name__ == "__main__":
    t = datetime.fromtimestamp(time.time())
    s = t.strftime("%Y-%m-%d_%H:%M")
    print(f'{s} - Starting Main...')
    insert_data(df = search_tweets(), db_file = 'traveltweets.db')
    t = datetime.fromtimestamp(time.time())
    s = t.strftime("%Y-%m-%d_%H:%M")
    print(f'{s} - Complete!')
