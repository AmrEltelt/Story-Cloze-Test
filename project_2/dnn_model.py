# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 18:44:54 2019

@author: Amr
"""
import numpy as np
import pandas as pd
import tensorflow as tf


def create_model(features):
    featureColumns = []
    for key in features.keys():
        featureColumns.append(tf.feature_column.numeric_column(key=key))
    model = tf.estimator.DNNClassifier(feature_columns=featureColumns,
                                        hidden_units=[3200, 1600, 800], 
                                        dropout=0.3,
                                        n_classes = 2,
                                        activation_fn=tf.nn.relu,
                                        optimizer='Adam',
                                        model_dir = 'tf_checkpoints')
    print("\nModel Loaded")
    return model

def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(buffer_size=len(features)).repeat().batch(batch_size)
    return dataset

def test_input_fn(features, labels, batch_size):
    features=dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset

def train(model, X_train, y_train, batch_size, epochs):
    print('\nTraining model..')
    n_steps = round(len(X_train)/batch_size*epochs)
    model.train(
            input_fn=lambda:train_input_fn(X_train, y_train, batch_size=batch_size),
            steps=n_steps)

def test(model, X_test, y_test):
    print('\nTesting model..')
    eval_result = model.evaluate(
            input_fn=lambda:test_input_fn(X_test, y_test, X_test.shape[0]))
    return eval_result

def predict(model, X_test):
    print('\nPredicting..')
    predictions = model.predict(
        input_fn=lambda:test_input_fn(X_test, labels=None, batch_size=X_test.shape[0]))
    results = pd.DataFrame(columns=['y', 'probability'])
    n = 0
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        if class_id == 0:
            class_id = 1
        else:
            class_id = 2
        results.loc[n] = [class_id, 100 * probability]
        n = n+1
    return results


