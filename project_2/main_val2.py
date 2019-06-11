# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:32:18 2019

"""
import tensorflow as tf
import pandas as pd
import numpy as np
import h5py

# load dataset embedded with skip_thoughts

# =============================================================================
# # Load train set incase h5 file was full
# file = h5py.File('data/train_embed.h5', 'r')
# X_train = file['X_train'][0:62000, 1:]
# X_train = np.reshape(X_train, (len(X_train), -1))
# X_train = pd.DataFrame(X_train)
# X_train.columns = X_train.columns.astype(str)
# train_stories = pd.read_csv('data/train_stories_neg.csv')
# y_train = train_stories['answer'].values
# y_train = y_train[0:62000].astype(np.int32)
# y_train = pd.DataFrame(y_train, columns=['y'])
# print('Train set loaded')
# =============================================================================
# Load val set
file = h5py.File('data/val_embed.h5', 'r')
X_val = file['X_val'][:, 1:]
X_val = np.reshape(X_val, (len(X_val), -1))
y_val = file['y_val'][:]

binary_verifiers = []
for value in y_val:
    if value == 1:
        binary_verifiers.append([1, 0])
    else:
        binary_verifiers.append([0, 1])  
y_val = binary_verifiers

# Load test set
file = h5py.File('data/test_embed.h5', 'r')
X_test = file['X_test'][:, 1:]
X_test = np.reshape(X_test, (len(X_test), -1))
y_test = file['y_test'][:]

binary_verifiers = []
for value in y_test:
    if value == 1:
        binary_verifiers.append([1, 0])
    else:
        binary_verifiers.append([0, 1])  
y_test = binary_verifiers


print("\nVal and Tests sets Loaded")


# Main

# buffer
#X = X_train
#Y = y_train
X = X_val
Y = y_val
X_test = X_test
Y_test = y_test

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 64
display_step = 1

# Network Parameters
n_hidden_1 = 3072 # 1st layer number of features
n_hidden_2 = 1024 # 2nd layer number of features
n_hidden_3 = 512 # 2nd layer number of features
n_input = 3*4800 # Number of feature
n_classes = 2 # Number of classes to predict


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X)/batch_size)
        X_batches = np.array_split(X, total_batch)
        Y_batches = np.array_split(Y, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))
    global result 
    result = tf.argmax(pred, 1).eval({x: X_test, y: Y_test})

# Write results into CSV file
import datetime
now = datetime.datetime.now()
submission = pd.DataFrame(result)
submission.to_csv('submission_'+ now.strftime("%Y%m%d%H%M") + ".csv",index=False, header=False)
print("\nPredicted endings saved")