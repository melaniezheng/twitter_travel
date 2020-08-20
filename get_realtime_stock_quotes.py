#!/Users/melaniezheng/opt/anaconda3/bin/python

import os
from sqlalchemy import create_engine
import datetime, time
import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries # stock 

api_key = os.environ.get('VANTAGE_API_KEY')

def get_quotes(ticker, interval = '5min'):
    '''get stock data from vantage api.
    input: ticker, interval i.e. 1 min, 5 min, 30 min, 60min
    output: dataframe with 6 columns date, open ,high low, close, volumn, pct_change'''
    ts = TimeSeries(key=api_key, output_format='pandas')
    stockdf, meta_data = ts.get_intraday(symbol=ticker, interval=interval, outputsize = 'full')
    stockdf['pct_change'] = stockdf['4. close'].pct_change()
    stockdf.reset_index(inplace=True)
    stockdf['time'] = stockdf.date.apply(lambda dt: dt.time())
    stockdf['day'] = stockdf.date.apply(lambda dt: dt.dayofweek)
    
    # remove lines where quotes are outside of the normal trading hours. Time Zone': 'US/Eastern'
    closingbell = datetime.datetime.strptime('16:00:00', '%H:%M:%S')
    closingbell = datetime.time(closingbell.hour, closingbell.minute)

    openingbell = datetime.datetime.strptime('09:30:00', '%H:%M:%S')
    openingbell = datetime.time(openingbell.hour, openingbell.minute)
    
    stockdf = stockdf[np.logical_and(stockdf.time < closingbell, stockdf.time > openingbell)]
    stockdf = stockdf[stockdf.day.isin([0,1,2,3,4])]
    stockdf = stockdf.drop(columns=['time','day'])
    # stockdf = stockdf.set_index('date')
    return stockdf

def insert_data(db_file, tickers):
    
    engine = create_engine(f'sqlite:///{db_file}', echo=False)
    
    for ticker in tickers:
        df = get_quotes(ticker, '5min')
        df_update = pd.read_sql_query(sql=f'SELECT * FROM {ticker};', con=engine, parse_dates='date')
        df_update.update(df)
        df_update.to_sql(ticker, con=engine, if_exists='replace', index=False)

        dup_rows = pd.merge(df, df_update, on ='date', how='inner')['date']
        new_data = df[~df.date.isin(dup_rows)]
        new_data.to_sql(ticker, con=engine, if_exists='append', index=False)
        print(f'{ticker} prices loaded successfully!')
        time.sleep(20)

if __name__ == "__main__":
    t = datetime.datetime.fromtimestamp(time.time())
    s = t.strftime("%Y-%m-%d_%H:%M")
    print(f'{s} - Starting Main...')
    dbfn = 'stocks.db'
    tickers = ['DAL','UAL','JETS']
    try:
        insert_data(dbfn, tickers)
    except Exception as e:
        print(e)
    finally:
        t = datetime.datetime.fromtimestamp(time.time())
        s = t.strftime("%Y-%m-%d_%H:%M")
        print(f'{s} - Complete!')
