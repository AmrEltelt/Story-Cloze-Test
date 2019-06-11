# -*- coding: utf-8 -*-
"""
Created on Wed May 22 21:03:56 2019

"""
import pandas as pd
import numpy as np
import random

# Load datasets
train_stories = pd.read_csv('data/train_stories.csv').drop(columns = ['storyid','storytitle'])
val_stories = pd.read_csv('data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv').drop(columns = ['InputStoryid'])
test_stories = pd.read_csv('data/test_for_report-stories_labels.csv').drop(columns = ['InputStoryid'])
names = ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'end1', 'end2', 'answer']


# Create negative ending
length = len(train_stories)
corps = np.ravel(train_stories.values[:,:4])

neg_ends = pd.DataFrame(np.random.choice(corps, length, replace=False))
labels = pd.DataFrame([random.randint(0, 1) for i in range(length)])

X = pd.concat([train_stories, neg_ends, labels], axis=1).values
for i in range(length):
    if X[i, 6] == 1:
        X[i, 4], X[i, 5] = X[i, 5], X[i, 4]

train_stories_neg = pd.DataFrame(X, columns=names)
Id = pd.Series(np.arange(len(train_stories_neg)), name='Id')
train_stories_neg = pd.concat([Id, train_stories_neg], axis=1, join_axes=[Id.index])
train_stories_neg.to_csv('data/train_stories_neg.csv',index=False)

X  = val_stories.values
for i in range(len(X)):
    if X[i, 6] == 1:
        X[i, 6] = 0
    else:
        X[i, 6] = 1

val_stories = pd.DataFrame(X, columns=names)
Id = pd.Series(np.arange(len(val_stories)), name='Id')
val_stories = pd.concat([Id, val_stories], axis=1, join_axes=[Id.index])
val_stories.to_csv('data/val_stories.csv',index=False)

X  = test_stories.values
for i in range(len(X)):
    if X[i, 6] == 1:
        X[i, 6] = 0
    else:
        X[i, 6] = 1

test_stories = pd.DataFrame(X, columns=names)
Id = pd.Series(np.arange(len(test_stories)), name='Id')
test_stories = pd.concat([Id, test_stories], axis=1, join_axes=[Id.index])
test_stories.to_csv('data/test_stories.csv',index=False)

