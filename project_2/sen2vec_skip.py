# -*- coding: utf-8 -*-
"""
Created on Wed May 29 23:47:19 2019

"""
import pandas as pd
import numpy as np
import h5py
from skip_thoughts import skipthoughts


def transform(dataset, encoder): 

    # Generate skip-thought embeddings
    stories = dataset.values[:, 0:4]
    endings = dataset.values[:, 4:6]
    n_stories = len(stories)
    sentences_encoded = np.empty((n_stories, 2, 4800))
    print("Generating skip-thoughts embeddings")
    stories_encoded = encoder.encode(stories, verbose=False)
    for i in range(endings.shape[1]):
        sentences_encoded[:, i] = encoder.encode(endings[:, i], verbose=False)
    sentences_encoded = np.reshape(sentences_encoded, (n_stories, -1))
    for i in range(n_stories):
        sentences_encoded[i] = np.tile(stories_encoded[i], 2) + sentences_encoded[i]
    features = sentences_encoded

    # Generate binary verifiers
    answers = dataset['AnswerRightEnding'].values 
    binary_verifiers = []
    for value in answers:
        if value == 1:
            binary_verifiers.append([1, 0])
        else:
            binary_verifiers.append([0, 1])  
    target = binary_verifiers
    
    return features, target


# Load datasets
train_stories = pd.read_csv('data/train_stories_neg.csv')
val_stories = pd.read_csv('data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv').drop(columns = ['InputStoryid'])

print("Loading skip-thoughts model for embedding")
skipthoughts_model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(skipthoughts_model)

print("Generating features array for train data")
X_train, Y_train = transform(train_stories, encoder)
hf = h5py.File('data/train_embed.h5', 'w') 
hf.create_dataset('X_train', data=X_train)
hf.create_dataset('Y_train', data=Y_train)
hf.close()

print("Generating features array for validation data")
X_val, Y_val = transform(val_stories, encoder)
hf = h5py.File('data/val_embed.h5', 'w') 
hf.create_dataset('X_val', data=X_val)
hf.create_dataset('Y_val', data=Y_val)
hf.close()
