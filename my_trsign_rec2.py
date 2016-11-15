#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 21:43:07 2016

@author: dajcs
"""

# Load pickled data
import pickle
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle


# TODO: fill this in based on where you saved the training and testing data
training_file = 'traffic_sign_data/train.p'
testing_file = 'traffic_sign_data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


### To start off let's do a basic data summary.

# TODO: number of training examples
n_train = len(X_train)  # 39209 = 116 * 338 + 1

# TODO: number of testing examples
n_test = len(X_test)  # 12630

# TODO: what's the shape of an image?
image_shape = X_test.shape[1:]   # (32,32,3)

# TODO: how many classes are in the dataset
n_classes = np.max(y_test) - np.min(y_test) + 1     # 43

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)



#for i in range(len(X_train)):
#    file = 'xy_train/train_{:05d}_{:02d}.png'.format(i, y_train[i])
#    plt.imsave(file, X_train[i])
#
#for i in range(len(X_test)):
#    file = 'xy_test/test_{:05d}_{:02d}.png'.format(i, y_test[i])
#    plt.imsave(file, X_test[i])
#




def plot_images(row, col, images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == row * col

    # Create figure with 3x6 sub-plots.
    # fig, ax = subplots(figsize=(18, 2))
    fig, axes = plt.subplots(row, col, figsize=(int(1.5*row),int(1.5*col)))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(image_shape))

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "cl: {0}".format(cls_true[i])
        else:
            xlabel = "cl: {0}, pr: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])


#X_train, y_train = shuffle(X_train, y_train)

images = []
imgcls = []

for i in range(n_train):
    if np.mean(X_train[i]) > 130 and y_train[i] not in imgcls:
        images.append(X_train[i])
        imgcls.append(y_train[i])


plot_images(6,7, images, imgcls)

# shuffle train
#X_train, y_train = shuffle(X_train, y_train)



def normalize_greyscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.01, 0.99]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # ToDo: Implement Min-Max scaling for greyscale image data
    a = 0.01
    b = 0.99
    x_min = np.min(image_data)
    x_max = np.max(image_data)
    return a + (image_data - x_min) * (b - a) / (x_max - x_min)


def normalize(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.01, 0.99]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # ToDo: Implement Min-Max scaling for greyscale image data
    a = 0.01
    b = 0.99
    norm_img = []
    for img in image_data:
        x_min = np.min(img)
        x_max = np.max(img)
        norm_img.append(a + (img - x_min) * (b - a) / (x_max - x_min))
    return np.array(norm_img)



train_features = normalize(X_train)
test_features = normalize(X_test)
is_features_normal = True

print('features normalized on 0.01 - 0.99 scale')

norm_images = normalize(images)
plot_images(6,7, norm_images, imgcls)



# Change to float32, we're working with float32 in tf
train_features = train_features.astype(np.float32)
test_features = test_features.astype(np.float32)



# Turn labels into numbers and apply One-Hot Encoding
encoder = LabelBinarizer()
encoder.fit(y_train)
train_labels = encoder.transform(y_train)
test_labels = encoder.transform(y_test)

# Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)
is_labels_encod = True

print('Labels One-Hot Encoded')



# Get randomized datasets for training and validation
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.05,
    random_state=832289)

print('Training features and labels randomized and split.')


import os

# Save the data for easy access
pickle_file = 'traf_sign_norm_1hot.pickle'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file...')
    try:
        with open('traf_sign_norm_1hot.pickle', 'wb') as pfile:
            pickle.dump(
                {
                    'train_dataset': train_features,
                    'train_labels': train_labels,
                    'valid_dataset': valid_features,
                    'valid_labels': valid_labels,
                    'test_dataset': test_features,
                    'test_labels': test_labels,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# Load the modules
import pickle
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Reload the data
pickle_file = 'traf_sign_norm_1hot.pickle'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  X_train = pickle_data['train_dataset']
  y_train = pickle_data['train_labels']
  X_valid = pickle_data['valid_dataset']
  y_valid = pickle_data['valid_labels']
  X_test = pickle_data['test_dataset']
  y_test = pickle_data['test_labels']
  del pickle_data  # Free up memory


print('Data and modules loaded.')



#tf.set_random_seed(42)

# neural network with 1 layer of 43 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 3072]        # 32*32*3
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [3072, 43]     b[43]
#   · · · · · · · ·                                              Y [batch, 43]

# The model is:
#
# Y = softmax( X * W + b)
#              X: matrix for [batch] color images of 32*32 pixels,
#                 flattened (there are [batch] images in a mini-batch)
#              W: weight matrix with 3072 lines and 43 columns
#              b: bias vector with 43 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with [batch] lines and 43 columns




# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 43])
# weights W[3072, 43]  # 32*32*3
W = tf.Variable(tf.zeros([3072, 43]))
# biases b[43]
b = tf.Variable(tf.zeros([43]))



# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
XX = tf.reshape(X, [-1, 3072])   # [batch,3072]

# Linear Function XW + b
logits = tf.matmul(XX, W) + b    # [batch,3072] x [3072,43] = [batch,43]

# Y - computed prediction
Y = tf.nn.softmax(logits)        # [batch,43]

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector
# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
# cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  # normalized for batches of 100 images,
                                                            # *10 because  "mean" included an unwanted division by 10
# the prediction Y might have 0 elements => log inifinite (NaN) => possible solutions below:
# cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y + 1e-10), reduction_indices=1)
# cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_)  # [batch]
# Training loss
loss = tf.reduce_mean(cross_entropy)

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

epochs = 20
batch_size = 300
learning_rate = 0.05

# Gradient Descent
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# init
init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

# The accuracy measured against the validation set
validation_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

batch_count = int(len(X_train)/batch_size)

for epoch_i in range(epochs):

    # Progress bar
    batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')

    # The training cycle
    for batch_i in batches_pbar:
        # Get a batch of training features and labels
        batch_start = batch_i*batch_size
        X_batch = X_train[batch_start:batch_start + batch_size]
#            batch_features = np.reshape(batch_features, [-1, 3072])
        y_batch = y_train[batch_start:batch_start + batch_size]

        # Run optimizer and get loss
        _, l = session.run(
            [optimizer, loss],
            feed_dict={X: X_batch, Y_: y_batch})

        # Log every 50 batches
        if batch_i % log_batch_step == 0:
            # Calculate Training and Validation accuracy
            training_accuracy = session.run(accuracy, feed_dict={X: X_train, Y_: y_train})
            validation_accuracy = session.run(accuracy, feed_dict={X: X_valid, Y_: y_valid})

            # Log batches
            previous_batch = batches[-1] if batches else 0
            batches.append(log_batch_step + previous_batch)
            loss_batch.append(l)
            train_acc_batch.append(training_accuracy)
            valid_acc_batch.append(validation_accuracy)

    # Check accuracy against Validation data
    validation_accuracy = session.run(accuracy, feed_dict={X: X_valid, Y_: y_valid})
    test_accuracy = session.run(accuracy, feed_dict={X: X_test, Y_: y_test})


loss_plot = plt.subplot(211)
loss_plot.set_title('Loss')
loss_plot.plot(batches, loss_batch, 'g')
loss_plot.set_xlim([batches[0], batches[-1]])
acc_plot = plt.subplot(212)
acc_plot.set_title('Accuracy')
acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
acc_plot.plot(batches, valid_acc_batch, 'b', label='Validation Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([batches[0], batches[-1]])
acc_plot.legend(loc='best')
plt.tight_layout()
plt.show()

print('Validation accuracy at {}'.format(validation_accuracy))
print('Test accuracy at {}'.format(test_accuracy))



























































