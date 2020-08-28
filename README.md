### Objective
Using travel tweets to predict and trave sector ETF 'JETS' price. Specifically, combine the current JETS price (at time t) and data extracted from twitter to generate a random forest regressor to predict next JETS (at time t+15 minuutes). 

__date range__: 2020-07-15 to 2020-07-27. <br>
__twitter data__: twitter API, tweepy <br>
__stock data__: Alpha Vantage API <br>
__python__ : version 3.7.4

Note that the training data range is only 2 weeks period and the project is just a POC as there are many limitations.

#### NLP on tweets:
- search keyword: travel 
- remove retweets --> save to sqlite database (cronjob every 5 minute)
- remove non-english tweets (due to googletrans api restriction on daily call limits)
- remove urls and mentions
- process emojis and non-text word representations
- cleaning tweets: lower case, tokenize, remove stopwords, lemmatize and stem words
- use TextBlob to get sentiment and polarity of each tweets

#### Stock price data:
- get realtime stock price (ticker:JETS) at 5 minute interval 
- stock data columns: datetime, open, high, low, close, volumn, pct_change
- save to sqlite database (cronjob every day)
- mask current close price (this will be the target of the predictive model) and shift close column to represent last_price(t-5 minutes) and last_volumn(t-5 minutes)

#### Predict JETS Price at t+ 15min using Feed Forward Neural Net:
- build neural net with 50 neurons in hidden layer. (try simple NN first before adding more hidden layers)
- feature preprocessing and engineering: 
  - hand craft features: tweets_count (in 15 minute interval), average sentiment, average polarity, average verified user, average followers count
  - total features 8 (above + last stock price, last trading volumn, last percent change)
  - normalize and scale features
- training:
  - evaluate model using Mean Squared Error and Mean Absolute Error
  - try batch norm
  - for more complex model, try dropout layers
  - 80/20 split for train/validation
  - early stopping to minimize validation loss
  - plot model training history to make sure model is not overfit or underfit
- prediction:
  - script to preprocess prediction data and generate feature vectors
  - prediction accuracy measured in Mean Absoute Error and time.

#### TO DO:
- write algorithm to output trading strategy (buy, sell, hold) that satisfied on investing objectives within constraints. 
- explore other models. especially RNN which makes sense for sequencial stock market.
- try different feature engineering. i.e. adding more text feature from tweets, try w2v model, n-gram tfidf.
- stream real-time tweet data (instead of api.search) for more up-to-date analysis and prediction.
- pay for googletrans api to make use of massive non-english tweets.
