# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 18:46:15 2019

"""

import pandas as pd
import numpy as np
import h5py
from dnn_model import *

# load dataset embedded with skip_thoughts

# =============================================================================
# # Load train set incase h5 file was full
# file = h5py.File('data/train_embed.h5', 'r')
# X_train = file['X_train'][:]
# X_train = np.reshape(X_train, (len(X_train), -1))
# X_train = pd.DataFrame(X_train)
# X_train.columns = X_train.columns.astype(str)
# train_stories = pd.read_csv('data/train_stories_neg.csv')
# y_train = train_stories['answer'].values
# y_train = y_train[:].astype(np.int32)
# y_train = pd.DataFrame(y_train, columns=['y'])
# # =============================================================================
# # # Load train set normal way
# # file = h5py.File('data/train_embed.h5', 'r')
# # X_train = pd.DataFrame(file['X_train_re'][:])
# # X_train.columns = X_train.columns.astype(str)
# # y_train = pd.DataFrame(file['y_train'][:].astype(np.int32), columns=['y'])
# # =============================================================================
# print('Train set loaded')
# =============================================================================
# Load val set
file = h5py.File('data/val_embed.h5', 'r')
X_val = pd.DataFrame(file['X_val_re'][:])
X_val.columns = X_val.columns.astype(str)
y_val = pd.DataFrame(file['y_val'][:].astype(np.int32), columns=['y'])
file = h5py.File('data/test_embed.h5', 'r')
# Load test set
X_test = pd.DataFrame(file['X_test_re'][:])
X_test.columns = X_test.columns.astype(str)
y_test = pd.DataFrame(file['y_test'][:].astype(np.int32), columns=['y'])
print("\nVal and Tests sets Loaded")


# Main

# buffer
features_train = X_val
label_train = y_val
features_test = X_test
label_test = y_test
features_predict = X_test
label_predict = y_test

epochs = 30
batch_size = 64

dnn = create_model(features_train)

train(dnn, features_train, label_train, batch_size, epochs)

accuracy = test(dnn, features_test, label_test)
print('\nTest accuracy: {accuracy:0.3f}'.format(**accuracy))

predictions = predict(dnn, features_predict)

# Write results into CSV file
import datetime
now = datetime.datetime.now()
submission = predictions.drop(columns = ['probability'])
submission.to_csv('submission_'+ now.strftime("%Y%m%d%H%M") + ".csv",index=False, header=False)
print("\nPredicted endings saved")



