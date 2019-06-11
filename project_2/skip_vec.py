# -*- coding: utf-8 -*-
"""
Created on Wed May 29 23:47:19 2019

"""
import pandas as pd
import numpy as np
import h5py
from skip_thoughts import skipthoughts


def skip_vectorize(dataset, encoder):
    # select features
    features_stories = dataset.values[:, 3:7]
    num_rows, num_cols = features_stories.shape
    
    print("Creating skip-thoughts vectors")
    features_encoded = np.empty((num_rows, num_cols, 4800))
    for i in range(num_cols):
        features_encoded[:, i] = encoder.encode(features_stories[:, i], verbose=False)
    features_encoded_re = np.reshape(features_encoded, (num_rows, -1))
    
    return features_encoded, features_encoded_re


# Load datasets
train_stories = pd.read_csv('data/train_stories_neg.csv')
val_stories = pd.read_csv('data/val_stories.csv')
test_stories = pd.read_csv('data/test_stories.csv')

skipthoughts_model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(skipthoughts_model)

X_train, X_train_re = skip_vectorize(train_stories, encoder)
y_train = train_stories['answer'].values

X_val, X_val_re = skip_vectorize(val_stories, encoder)
y_val = val_stories['answer'].values

X_test, X_test_re = skip_vectorize(test_stories, encoder)
y_test = test_stories['answer'].values


hf = h5py.File('data/train_embed.h5', 'w') 
hf.create_dataset('X_train', data=X_train)
#hf.create_dataset('X_train_re', data=X_train_re)
#hf.create_dataset('y_train', data=y_train)
hf.close()
hf = h5py.File('data/val_embed.h5', 'w') 
hf.create_dataset('X_val', data=X_val)
hf.create_dataset('X_val_re', data=X_val_re)
hf.create_dataset('y_val', data=y_val)
hf.close()
hf = h5py.File('data/test_embed.h5', 'w') 
hf.create_dataset('X_test', data=X_test)
hf.create_dataset('X_test_re', data=X_test_re)
hf.create_dataset('y_test', data=y_test)
hf.close()


