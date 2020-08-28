import datetime
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_validate
import logging
# neural nets
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import matplotlib.dates as mdates
register_matplotlib_converters()

import preprocessing

def plot_model_loss(history, fn):
    fig = plt.figure(figsize = (8,4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(fn)

def fit_and_score(model, te_pred, X_tr, y_tr, X_te, y_te, test_datetime, plot=True, fn = None):
    if model is None:
        if te_pred is not None:
            print(f'Test MAE: {mean_absolute_error(y_te, te_pred)}')
        else:
            raise ValueError('te_pred is None. Provide test prediction value!')
    else:
        cv_score = cross_validate(model, X_tr, y_tr, scoring = 'neg_mean_absolute_error', cv = 3, return_train_score = True)
        print('Cross-Validation Scores:')
        print("Train MAE: {m} +/- {sd}".format(m = round(-np.mean(cv_score['train_score']),5), sd = round(np.std(cv_score['train_score']),4)))
        print("Eval MAE: {m} +/- {sd}".format(m = round(-np.mean(cv_score['test_score']),5), sd = round(np.std(cv_score['test_score']),4)))
        model.fit(X_tr, y_tr)
        print(f'- '*15)
        te_pred = model.predict(X_te)
        print(f'Test MAE: {mean_absolute_error(y_te, te_pred)}')
        
    if plot:
        fig = plt.figure(figsize = (8,4))
        ax = plt.gca()
        formatter = mdates.DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(formatter)
        locator = mdates.HourLocator()
        ax.xaxis.set_major_locator(locator)
        dt = pd.to_datetime(test_datetime)
        plt.title(f'{type(model).__name__}')
        plt.plot(test_datetime,y_te)
        plt.plot(test_datetime,te_pred)
        plt.legend(['actual','prediction'])
        plt.title('feed forward Neural Net - 50 neurons')
        if fn is not None:
            plt.savefig(fn)       

def ffnn(num_layers, neurons):
    '''build ffnn
    input: 
    num_layers, integer, 
    neurons, list of integers
    output:
    modl'''
    model = Sequential()
    for l in range(num_layers):
        model.add(Dense(neurons[l]))
    model.add(Dense(1))
    adam = optimizers.Adam(learning_rate = params['lr'])
    model.compile(optimizer=adam,loss='mean_squared_error')

    return model

if __name__ == "__main__":
    print('training neural network ...')
    # features are already scaled during preprocessing.
    X_train, y_train, X_test, y_test, train_dt, test_dt = preprocessing.run_preprocessor(type = "training", 
                                                                                        tweets_file = 'tweets_2020-07-15_2020-07-27.csv', 
                                                                                        stocks_file = 'JETS_2020-07-15_2020-07-27.csv')
    print('loaded feature vectors successfully')
    params = {
        "batch_size": 5,  # 20<16<10, 25 was a bust
        "epochs": 500,
        "lr": 0.050000,
        "time_steps": 4
    }

    TIME_STEPS = params["time_steps"]
    BATCH_SIZE = params['batch_size']

    ffnn50 = ffnn(1,[50])
    es = EarlyStopping(monitor='val_loss', mode='min', patience=7)
    cp = ModelCheckpoint(filepath='./models/checkpoints/ffnn50.{epoch:02d}-{val_loss:.2f}.h5',monitor='val_loss', \
        verbose=1,save_best_only=True, save_weights_only=False, mode='min')
    my_callbacks = [es,cp]

    print(X_train.shape)
    print(X_test.shape)
    history = ffnn50.fit(X_train,y_train.values,epochs=params['epochs'],batch_size=params['batch_size'], 
                            validation_split=0.2, callbacks=my_callbacks)
    print("saving model...")
    pickle.dump(ffnn50, open("./models/ffnn50", "wb"))

    # plot train and validation loss
    plot_model_loss(history, 'ffnn50_model_loss.png')

    # predict test set
    y_pred = ffnn50.predict(X_test)
    estimator = None
    # plot actual vs prediction
    fit_and_score(estimator, y_pred, X_train, y_train, X_test, y_test, test_dt, fn = './models/plots/ffnn50_test_accuracy.png')