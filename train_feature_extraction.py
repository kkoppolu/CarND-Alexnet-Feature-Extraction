import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np
import time

# TODO: Load traffic signs data.
training_file = 'train.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
X_all, y_all = train['features'], train['labels']
nb_classes = np.unique(y_all).size

# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, test_size=0.2)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
onehot_y = tf.one_hot(y, nb_classes)

resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)

fc8w = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8w, fc8b)
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
# get softmax probabilities of the scores
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_y)
loss_operation = tf.reduce_mean(cross_entropy)
train_opt = tf.train.AdamOptimizer().minimize(loss_operation, var_list=[fc8w, fc8b])

# get the class predictions from the logits
class_predicted = tf.argmax(probs, 1)
# get the actual class from the labels
actual_class = tf.argmax(onehot_y,1)
# Check for correct model predictions
correct_prediction = tf.equal(class_predicted, actual_class)
# Measure accuracy as the average number of correct predictions
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: Train and evaluate the feature extraction model.
"""Evaluate the Conv Net model

Keyword arguments:
x_set -- Data set to evaluate against
y_set -- Labels of the data set
"""
def evaluate(x_set, y_set):
    sess = tf.get_default_session()
    
    num_examples = len(x_set)
    num_correct_preds = 0.0
    total_loss=0.0
    for i in range(0, num_examples, batch_size):
        start_idx = i
        end_idx = i + batch_size
        x_batch = x_set[start_idx:end_idx]
        y_batch = y_set[start_idx:end_idx]
        loss, accuracy = sess.run([loss_operation, accuracy_operation], 
                                  feed_dict= {x: x_batch, y:y_batch})
        num_correct_preds += (len(x_batch) * accuracy)
        total_loss += (loss * len(x_batch))
    return total_loss/num_examples, num_correct_preds/num_examples


from sklearn.utils import shuffle

batch_size = 128 # use power of 2 for GPU optimization
epochs = 10

num_examples = len(X_train)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Starting training")
    print ("Number of examples: {}".format(num_examples))

    for j in range(epochs):
        t0 = time.time()
        print ("Epoch {} ...".format(j+1))
        X_train, y_train = shuffle(X_train, y_train)

        for i in range(0, num_examples, batch_size):
            start_idx = i
            end_idx = i + batch_size
            x_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            sess.run(train_opt, feed_dict= {x: x_batch, y: y_batch})

        training_loss, training_accuracy = evaluate(X_train, y_train)
        print ("Epoch: {}: Training Loss: {:.3f}".format(j+1, training_loss))
        print ("Epoch: {}: Training Accuracy: {:.3f}".format(j+1, training_accuracy))
        print ("Epoch: {}: Time: {:.3f}".format(j+1, time.time() - t0))

        validation_loss, validation_accuracy = evaluate(X_valid, y_valid)
        print ("Epoch: {}: Validation Loss: {:.3f}".format(j+1, validation_loss))  
        print ("Epoch: {}: Validation Accuracy: {:.3f}".format(j+1, validation_accuracy))

