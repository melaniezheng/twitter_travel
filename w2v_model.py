import os
import pandas as pd
import numpy as np
import joblib

# W2V library
import gensim
from gensim.models import word2vec, KeyedVectors
import logging

def word2vec_model(list_of_texts, model_name, num_features = 200):
    # train word2vec model
    ''' default setting
    class gensim.models.word2vec.Word2Vec(sentences=None, corpus_file=None, size=100, 
                                          alpha=0.025, window=5, min_count=5, 
                                          max_vocab_size=None, sample=0.001, seed=1, workers=3, 
                                          min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75, 
                                          cbow_mean=1, hashfxn=<built-in function hash>, iter=5, 
                                          null_word=0, trim_rule=None, sorted_vocab=1, 
                                          batch_words=10000, compute_loss=False, callbacks=(), 
                                          max_final_vocab=None '''
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Creating the model and setting values for the various parameters
    # we can make these variables inputs to the function if needed.

    num_features = num_features  # Word vector dimensionality
    min_word_count = 5 # Minimum word count
    num_workers = 4     # Number of parallel threads
    context = 5        # Context window size
    downsampling = 1e-5 # (0.01) Downsample setting for frequent words
    model = word2vec.Word2Vec(list_of_texts,
                              workers=num_workers,
                              size=num_features,
                              min_count=min_word_count,
                              window=context,
                              sample=downsampling)

    # Save Model
    model.init_sims(replace=True) # memory efficient
    model.save(model_name)
    return model

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
    try:
        df = pd.from_csv(f'./data/tweets_{from_date}_{to_date}.csv')
    except:
        raise FileNotFoundError("Please make sure file exists in data folder!\
            If file is not found, run clean_tweets_data.py to save a file for specified dates.")
    print("Creating Word2Vec Model...")
    model_name = input('please enter model name to save')
    word2vec_model(df.clean_text.tolist(), model_name)

    
    