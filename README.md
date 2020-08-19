### Objective
Using travel tweets to predict and trave sector ETF 'JETS' price. Specifically, combine the current JETS price (at time t) and data extracted from twitter to generate a random forest regressor to predict next JETS (at time t+5 minuutes). 

Note that although the model is able to predict the next stock price at the high accuracy, it is very difficult to monetize this due to the day-trading restrictions for trading in high frequency and associated transaction cost. 

__date range__: 2020-07-15 to 2020-07-27. <br>
__twitter data__: twitter API, tweepy <br>
__stock data__: Alpha Vantage API <br>


#### NLP on tweets:
- search keyword: travel 
- remove retweets --> save to sqlite database (cronjob every 5 minute)
- remove non-english tweets (due to googletrans api restriction on daily call limits)
- remove urls and mentions
- process emojis and non-text word representations
- cleaning tweets: lower case, tokenize, remove stopwords, lemmatize and stem words
- gensim's w2v model and sklearn's tfidf model to vectorize cleaned tweets

#### Stock price data:
- get realtime stock price (ticker:JETS) at 5 minute interval 
- stock data columns: datetime, open, high, low, close, volumn, pct_change
- save to sqlite database (cronjob every day)
- mask current close price (this will be the target of the predictive model) and shift close column to represent last_price(t-5 minutes) and last_volumn(t-5 minutes)

#### Random Forest Regressor (RFR):
- train and save gensim's w2v and tfidf model
- build pipeline for preprocessing
- 75 / 25 train/test split
- 3 fold cross validation for hyperparameter tuning for RFR
- finalize and save RFR model
- "productionize" and run a simulation with brand new test data (out of bag).

#### TO DO:
- explore other models. especially RNN which makes sense for sequencial stock market.
- try different feature engineering. i.e. adding more text feature from tweets, different w2v model, n-gram tfidf.
- stream real-time tweet data (instead of api.search)
- pay for googletrans api to make use of massive non-english tweets.
