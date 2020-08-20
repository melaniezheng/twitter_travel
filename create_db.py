import os
import sys
import sqlite3
import time
from datetime import datetime
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f'Connection Successful! {sqlite3.version}')
        return conn
    except Error as e:
        print(e)
    

def create_tables(db_file, tablename, sql_cmd):
    #Creating TRAVEL table
    conn = create_connection(db_file)
    for tn in tablename:
        cursor = conn.cursor()
        drop = f'DROP TABLE IF EXISTS {tn};'
        if sql_cmd == 'tweets':
            sql = f'''CREATE TABLE {tn}(id_str TEXT NOT NULL PRIMARY KEY,
               datetime INTEGER NOT NULL,
               username TEXT,
               verified INTEGER,
               followers_count INTEGER,
               friends_count INTEGER,
               user_location TEXT,
               coordinates BLOB,
               place TEXT,
               country TEXT,
               place_coordinates BLOB,
               text BLOB NOT NULL,
               num_retweets INTEGER,
               num_likes INTEGER
            )'''
        if sql_cmd == 'stocks':
            sql = f'''CREATE TABLE {tn}(date DATETIME NOT NULL PRIMARY KEY,
            "1. open" FLOAT,
            "2. high" FLOAT,
            "3. low" FLOAT,
            "4. close" FLOAT NOT NULL,
            "5. volume" FLOAT NOT NULL,
            pct_change FLOAT
            )'''
        cursor.execute(drop)
        cursor.execute(sql)
        print(f"Table {tn} created successfully........")

        # Commit your changes in the database
        conn.commit()

    #Closing the connection
    conn.close()

if __name__ == "__main__":
    db_file = input('please enter database filename: ')
    tablename = input('please enter new table name(s): (comma seperated when multiple)')
    sql_cmd = input('tweets or stocks?')
    print(f'Creating Database...')
    try:
        create_tables(db_file, tablename.split(','), sql_cmd)
    except Exception as e:
        print(e)
        print('operation failed')