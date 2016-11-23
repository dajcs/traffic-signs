#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:18:06 2016

@author: dajcs
"""


# Load the modules
import pickle
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Reload the data
pickle_file = 'traf_sign_norm_rand_1hot.pickle'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    X_train = pickle_data['X_train']
    y_train = pickle_data['y_train']
    X_test = pickle_data['X_test']
    y_test = pickle_data['y_test']
    del pickle_data  # Free up memory


print('Training and testing data is loaded.')



# neural network structure 3 * cnn + 2 * sigmoid + 1 * softmax:
#
# · · · · · · · · · ·      (input data, 3-deep(color))           X [batch, 32, 32, 3]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 6x6x3=>18 stride 1        W1 [6, 6, 3, 18]        B1 [18]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                            Y1 [batch, 32, 32, 36]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x18=>36 stride 2       W2 [5, 5, 18, 36]       B2 [36]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶                                              Y2 [batch, 16, 16, 36]
#     @ @ @ @ @ @       -- conv. layer 4x4x36=>72 stride 2       W3 [4, 4, 36, 72]       B3 [72]
#     ∶∶∶∶∶∶∶∶∶∶                                                Y3 [batch, 8, 8, 72] => reshaped to YY [batch, 8*8*72]
#      \x/x\x\x/        -- fully connected layer (sigmoid)       W4 [8*8*72, 800]        B4 [800]
#       · · · ·                                                  Y4 [batch, 800]
#       \x/x\x/         -- fully connected layer (sigmoid)       W5 [800, 200]           B5 [200]
#       · · · ·                                                  Y5 [batch, 200]
#        \\x//          -- fully connected layer (softmax)       W6 [200, 43]            B6 [43]
#        · · ·                                                   Y [batch, 43]

tf.set_random_seed(3333)

# input X: 32*32 color images, the first dimension (None) will index the images
# in the mini-batch
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 43])
# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.6 at training time
pkeep = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 43 softmax neurons)
K = 18   # first convolutional layer output depth
L = 36   # second convolutional layer output depth
M = 72   # third convolutional layer
N = 800  # fully connected layer
O = 200  # fully connected layer

# relu Bias needs to be initialized to a small number > 0
W1 = tf.Variable(tf.truncated_normal([6, 6, 3, K], stddev=0.1))  # 6x6 patch, 3 input channel, K output channels (32)
B1 = tf.Variable(tf.ones([K])/22)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))  # 5x5 patch, K input channel, L output channels (64)
B2 = tf.Variable(tf.ones([L])/22)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))  # 4x4 patch, L input channel, M output channels (128)
B3 = tf.Variable(tf.ones([M])/22)
W4 = tf.Variable(tf.truncated_normal([8 * 8 * M, N], stddev=0.1))  # fully conn [8*8*M,N] -> (3072,800)
B4 = tf.Variable(tf.zeros([N]))
W5 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))        # fully conn [N, O]  [800,200]
B5 = tf.Variable(tf.zeros([O]))
W6 = tf.Variable(tf.truncated_normal([O, 43], stddev=0.1))       # fully conn [O, 43] -> (200,43)
B6 = tf.Variable(tf.zeros([43]))


# The model
stride = 1  # output is 32*32
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)  #[batch, 32, 32, 36]
stride = 2  # output is 16*16
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 8*8
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 8 * 8 * M])   # [batch, 3072]

Y4 = tf.nn.sigmoid(tf.matmul(YY, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)
Y5 = tf.nn.sigmoid(tf.matmul(Y4d, W5) + B5)
Y5d = tf.nn.dropout(Y5, pkeep)

Ylogits = tf.matmul(Y5d, W6) + B6
Y = tf.nn.softmax(Ylogits)


# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)
cross_entropy = tf.reduce_mean(cross_entropy)


# Training loss
loss = tf.reduce_mean(cross_entropy)

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



pk = 0.6
epochs = 5
batch_size = 300
learning_rate = 0.003
regularization_fix = 13

# AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# saver
saver = tf.train.Saver()

# init
init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

# Measurements use for graphing loss and accuracy
log_batch_step = 20
batches = []
loss_batch = []
train_acc_batch = []
test_acc_batch = []

batch_count = int(len(X_train)/batch_size)

for epoch_i in range(epochs):

    # Progress bar
    batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')

    # The training cycle
    for batch_i in batches_pbar:
        # Get a batch of training features and labels
        batch_start = batch_i*batch_size
        X_batch = X_train[batch_start:batch_start + batch_size]
        y_batch = y_train[batch_start:batch_start + batch_size]

        # compute loss
        l = session.run(loss,
            feed_dict={X: X_batch, Y_: y_batch, pkeep: 1.0})

        # fixing regularization
        tf.set_random_seed(((epoch_i*batch_count)+batch_i)%regularization_fix)

        # optimize parameters
        _ = session.run(optimizer,
            feed_dict={X: X_batch, Y_: y_batch, pkeep: pk,
                       lr: learning_rate * ((epochs - epoch_i)/epochs)})

        # Log every [step] batches
        if batch_i % log_batch_step == 0:
            # Calculate Training and Validation accuracy
            training_accuracy = session.run(accuracy, feed_dict={X: X_train[:11111],
                                                                 Y_: y_train[:11111], pkeep: 1.0})
            test_accuracy = session.run(accuracy, feed_dict={X: X_test[:11111], Y_: y_test[:11111], pkeep: 1.0})

            # Log batches
            previous_batch = batches[-1] if batches else 0
            batches.append(log_batch_step + previous_batch)
            loss_batch.append(l)
            train_acc_batch.append(training_accuracy)
            test_acc_batch.append(test_accuracy)
#    print('\n','test_accuracy=',test_accuracy)

test_accuracy = session.run(accuracy, feed_dict={X: X_test[:11111], Y_: y_test[:11111], pkeep: 1.0})

# checkpoint save
saver.save(session, '3cnn_2sigmoid_1softmax2.ckpt') # 1: 957, 3: 956, 2:955
session.close()

loss_plot = plt.subplot(211)
loss_plot.set_title('Loss')
loss_plot.plot(batches, loss_batch, 'g')
loss_plot.set_xlim([batches[0], batches[-1]])
loss_plot.set_yscale('log')
acc_plot = plt.subplot(212)
acc_plot.set_title('Accuracy')
acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
acc_plot.plot(batches, test_acc_batch, 'c', label='Test Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([batches[0], batches[-1]])
acc_plot.legend(loc='lower right')
plt.tight_layout()
plt.show()

print('epochs =', epochs) # 33
print('batch_size =',batch_size) # 300
print('learning_rate =',learning_rate) # 0.0005
print('pkeep =', pk)
print('regularization_fix =',regularization_fix)
print('Last Test accuracy: {}'.format(test_accuracy))
print('Max Test accuracy: {}'.format(np.max(test_acc_batch)))

#epochs = 5
#batch_size = 300
#learning_rate = 0.003
#pkeep = 0.6
#Last Test accuracy: 0.957249641418457
#Max Test accuracy: 0.9584197402000427
#tf.set_random_seed(3333)
#B1 = tf.Variable(tf.ones([K])/128)
#B2 = tf.Variable(tf.ones([L])/32)
#B3 = tf.Variable(tf.ones([M])/8)
#B4 = tf.Variable(tf.zeros([N])/430)
#B5 = tf.Variable(tf.zeros([43])/43)
#tf.set_random_seed(((epoch_i*batch_count)+batch_i)%13)
