import os
import sys
import pandas as pd
import numpy as np
import sqlite3
import time
from datetime import datetime
from sqlalchemy import create_engine

def get_stock_data(ticker, from_date, to_date, db_file = 'stocks.db'):
    '''get stock intra day quotes from sqllite database, 
    clean stock data: 
    1. remove outside of normal trading hour, M-F 9:30 - 16:00 EST,
    2. shift rows to represent last volumn, last close price for Machine Learning model,
    3. drop columns not used by the ML model: open, high, low
    output: cleaned df
    '''
    engine = create_engine(f'sqlite:///{db_file}', echo=False) 
    data = pd.read_sql_query(sql=f'SELECT * FROM {ticker} WHERE date >= {from_date} and \
        date <= {to_date};', con=engine)
    data.drop(columns=['1. open','2. high','3. low'], inplace = True)
    data.columns=['close','volumn','pct_change']
    target = data['close']
    stockdf = data.shift(-1).rename(columns={'close':'last_close','volumn':'last_vol', 'pct_change':'last_pct_change'})
    stockdf['target'] = target
    stockdf.reset_index(inplace=True)
    stockdf['time'] = stockdf.date.apply(lambda dt: dt.time())
    stockdf['day'] = stockdf.date.apply(lambda dt: dt.dayofweek)
    
    # remove lines where quotes are outside of the normal trading hours. Time Zone': 'US/Eastern'
    closingbell = datetime.datetime.strptime('16:00:00', '%H:%M:%S')
    closingbell = datetime.time(closingbell.hour, closingbell.minute)

    openingbell = datetime.datetime.strptime('09:30:00', '%H:%M:%S')
    openingbell = datetime.time(openingbell.hour, openingbell.minute)
    
    stockdf = stockdf[np.logical_and(stockdf.time <= closingbell, stockdf.time >= openingbell)]
    stockdf = stockdf[stockdf.day.isin([0,1,2,3,4])] # Monday to Friday
    stockdf.drop(columns=['time','day'], inplace=True)

    stockdf.to_csv(f'./data/{ticker}_{from_date}-{to_date}.csv', index=False)

def validate(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")

if __name__ == '__main__':
    try:
        ticker = sys.argv[1]
        print(f'Entered ticker: {ticker}')
    except IndexError:
        raise IndexError('Oops, not a valid ticker! Try again... i.e. JETS, UAL, DAL')
    try:
        from_date = sys.argv[2]
        to_date = sys.argv[3]
    except IndexError:
        raise IndexError('Please provide a from_date and to_date, format YYYY-MM-DD')
    validate(from_date)
    validate(to_date)
    get_stock_data(ticker, from_date, to_date)
    print('Done!')


    
    
