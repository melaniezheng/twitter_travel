import os
import datetime
import pandas as pd
import numpy as np
import joblib
import pickle
import preprocessing 


def predict(tweets_file, stocks_file, model_name):
    print('Running Preprocessor...')
    X, dt = preprocessing.run_preprocessor("predicting", tweets_file, stocks_file, interval = 15)
    print('Loading Model...')
    try:
        model = pickle.load(open(f'./models/{model_name}', 'rb'))
        print("Loaded model successfully!")
    except FileNotFoundError:
        print("Model not found")
        
    pred = model.predict(X)
    return (pred, dt)
    

if __name__ == "__main__":
    tweets_file = input('Enter tweets filename: ')
    stocks_file = input('Enter stocks filename: ')
    model_name = input('Enter model name: ')
    prediction, dt = predict(tweets_file, stocks_file, model_name)
    print(f'{dt}-Prediction-{prediction}')