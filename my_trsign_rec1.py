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

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Reload the data
pickle_file = 'traf_sign_norm_1hot.pickle'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  train_features = pickle_data['train_dataset']
  train_labels = pickle_data['train_labels']
  valid_features = pickle_data['valid_dataset']
  valid_labels = pickle_data['valid_labels']
  test_features = pickle_data['test_dataset']
  test_labels = pickle_data['test_labels']
  del pickle_data  # Free up memory


print('Data and modules loaded.')








features_count = 32 * 32 * 3   # 3072
labels_count = 43

# Set the features and labels tensors (placeholder)
features = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

# Set the weights and biases tensors
weights = tf.Variable(tf.truncated_normal([features_count,labels_count]))
biases = tf.Variable(tf.zeros([labels_count]))

from tensorflow.python.ops.variables import Variable

assert features._op.name.startswith('Placeholder'), 'features must be a placeholder'
assert labels._op.name.startswith('Placeholder'), 'labels must be a placeholder'
assert isinstance(weights, Variable), 'weights must be a TensorFlow variable'
assert isinstance(biases, Variable), 'biases must be a TensorFlow variable'

assert features._shape == None or (\
    features._shape.dims[0].value is None and\
    features._shape.dims[1].value in [None, 3072]), 'The shape of features is incorrect'
assert labels._shape in [None, 43], 'The shape of labels is incorrect'
assert weights._variable._shape == (3072, 43), 'The shape of weights is incorrect'
assert biases._variable._shape == (43), 'The shape of biases is incorrect'

assert features._dtype == tf.float32, 'features must be type float32'
assert labels._dtype == tf.float32, 'labels must be type float32'




# Feed dicts for training, validation, and test session
train_features = np.reshape(train_features, [-1, 3072])
valid_features = np.reshape(valid_features, [-1, 3072])
test_features = np.reshape(test_features, [-1, 3072])
train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}
#train_feed_dict = {features: train_features.reshape(-1,3072), labels: train_labels}
#valid_feed_dict = {features: valid_features.reshape(-1,3072), labels: valid_labels}
#test_feed_dict = {features: test_features.reshape(-1,3072), labels: test_labels}

# Linear Function WX + b
logits = tf.matmul(features, weights) + biases

prediction = tf.nn.softmax(logits)

# Cross entropy
cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)

# Training loss
loss = tf.reduce_mean(cross_entropy)

# Create an operation that initializes all variables
init = tf.initialize_all_variables()

# Test Cases
with tf.Session() as session:
    session.run(init)
    session.run(loss, feed_dict=train_feed_dict)
    session.run(loss, feed_dict=valid_feed_dict)
    session.run(loss, feed_dict=test_feed_dict)
    biases_data = session.run(biases)

assert not np.count_nonzero(biases_data), 'biases must be zeros'

print('Tests Passed!')


# Determine if the predictions are correct
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
# Calculate the accuracy of the predictions
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

print('Accuracy function created.')





epochs = 5
batch_size = 100
learning_rate = 0.1



### DON'T MODIFY ANYTHING BELOW ###
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# The accuracy measured against the validation set
validation_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

with tf.Session() as session:
    session.run(init)
#    batch_count = int(math.ceil(len(train_features)/batch_size))
    batch_count = int(len(train_features)/batch_size)

    for epoch_i in range(epochs):

        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')

        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
#            batch_features = np.reshape(batch_features, [-1, 3072])
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer and get loss
            _, l = session.run(
                [optimizer, loss],
                feed_dict={features: batch_features, labels: batch_labels})

            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)

        # Check accuracy against Validation data
        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

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












































import tensorflow as tf

img_size_flat = image_shape[0] * image_shape[1] * image_shape[2]  # 32 * 32 * 3 = 3072
num_classes = n_classes

# placeholder ~ data
x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

# variable - trainable
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

# model
logits = tf.matmul(x, weights) + biases  # [batch,img_size] x [img_size, num_class] + [num_class] (broadcast)
y_pred = tf.nn.softmax(logits)              #  [batch, num_class]
y_pred_cls = tf.argmax(y_pred, dimension=1) #  [batch]

# cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)  # [batch]
cost = tf.reduce_mean(cross_entropy)   # scalar

# optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# performance monitoring
correct_prediction = tf.equal(y_pred_cls, y_true_cls)   # [batch]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # scalar, float32



# create tf session
session = tf.Session()

# init variables
session.run(tf.initialize_all_variables())




batch_size = 100


def optimize(num_iterations):
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)










































































