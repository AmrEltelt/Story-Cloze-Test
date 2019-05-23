# -*- coding: utf-8 -*-
"""
Created on Wed May 22 21:03:56 2019

"""
import pandas as pd
import numpy as np
import random

# Load datasets
train_stories = pd.read_csv('data/train_stories.csv').drop(columns = ['storyid','storytitle'])
eval_stories = pd.read_csv('data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv').drop(columns = ['InputStoryid'])

# Create negative ending
length = len(train_stories)
corps = np.ravel(train_stories.values[:,:4])

neg_ends = pd.DataFrame(np.random.choice(corps, length, replace=False))
labels = pd.DataFrame([random.randint(1, 2) for i in range(length)])

X = pd.concat([train_stories, neg_ends, labels], axis=1).values
for i in range(length):
    if X[i, 6] == 2:
        X[i, 4], X[i, 5] = X[i, 5], X[i, 4]

train_stories_neg = pd.DataFrame(X, columns=eval_stories.columns)
train_stories_neg.to_csv('data/train_stories_neg.csv',index=False)

