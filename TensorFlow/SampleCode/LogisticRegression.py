import os
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import xlrd
import tempfile
import urllib
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

# used for argument parse
FLAGS = None

def extract_samples_Fn(data, num_classes):
    index_list = []
    for sample_index in range(data.shape[0]):
        label = data[sample_index]
        if label < num_classes:
            index_list.append(sample_index)
    return index_list

def process_data(num_classes):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    index_list_train = extract_samples_Fn(y_train, num_classes)
    index_list_test = extract_samples_Fn(y_test, num_classes)
    data = {}
    data['train/image'] = x_train[index_list_train].reshape([x_train[index_list_train].shape[0], -1])
    data['train/label'] = y_train[index_list_train]
    data['test/image'] = x_test[index_list_test].reshape([x_train[index_list_test].shape[0], -1])
    data['test/label'] = y_test[index_list_test]
    return data

def logistic_model(input,num_classes):
    """
    Forward passing the X.
    :param X: Input.
    :return: X*W + b.
    """
    # A simple fully connected with two class and a softmax is equivalent to Logistic Regression.
    logits = tf.contrib.layers.fully_connected(inputs=input, num_outputs = num_classes, scope='fc')
    return logits

def logit_loss(logits, label_one_hot):
    """
    compute the loss by comparing the predicted value to the actual label.
    :param X: The output from model.
    :param Y: The label.
    :return: The loss over the samples.
    """
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label_one_hot))
    return loss

# The training function.
def train_helper(loss, num_train_samples, batch_size, num_epochs_per_decay, learning_rate_decay_factor, initial_learning_rate = 0.0001):
    with tf.name_scope('train_helper'):
        with tf.variable_scope('global'):
            global_step = tf.get_variable("global_step", initializer=0.0, trainable=False)
        decay_steps = int(num_train_samples / batch_size * num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                                global_step,
                                                decay_steps,
                                                learning_rate_decay_factor,
                                                staircase=True,
                                                name='exponential_decay_learning_rate')
    return learning_rate

def train(loss, learning_rate = 0.0001):
    with tf.name_scope('train_op'):
        with tf.variable_scope('global', reuse=True):
            global_step = tf.get_variable("global_step")
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients_and_variables = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(gradients_and_variables, global_step=global_step)
    return train_op

def check_accuracy(logits, label_one_hot):
    with tf.name_scope('accuracy'):
        prediction_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))
    return accuracy

def main(_):
    # get data
    data = process_data(FLAGS.num_classes)

    # Dimentionality of train
    dimensionality_train = data['train/image'].shape
    # Dimensions
    num_train_samples = dimensionality_train[0]
    num_features = dimensionality_train[1]

    # build graph
    important = tf.Graph()
    with important.as_default():
        # input data placeholder 
        image_place = tf.placeholder(tf.float32, shape=([None, int(num_features)]), name='image')
        label_place = tf.placeholder(tf.int32, shape=([None,]), name='gt')
        label_one_hot = tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1)
        dropout_param = tf.placeholder(tf.float32)
        # model
        logits = logistic_model(image_place, FLAGS.num_classes)
        # loss
        loss = logit_loss(logits, label_one_hot)
        # train_op
        learning_rate = train_helper(loss, num_train_samples, FLAGS.batch_size, FLAGS.num_epochs_per_decay, FLAGS.learning_rate_decay_factor, FLAGS.initial_learning_rate)
        train_op = train(loss, learning_rate)
        # Evaluate the model
        accuracy = check_accuracy(logits, label_one_hot)

    # The prefix for checkpoint files
    checkpoint_prefix = 'model'

    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
    with tf.Session(graph=important, config=session_conf) as sess:
        # # The saver op. TODO
        # saver = tf.train.Saver()

        # # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # # If fie-tuning flag in 'True' the model will be restored.
        # if FLAGS.fine_tuning:
        #     saver.restore(sess, os.path.join(FLAGS.log_dir, checkpoint_prefix))
        #     print("Model restored for fine-tuning...")

        for epoch in range(FLAGS.num_epochs):
            total_batch_training = int(data['train/image'].shape[0] / FLAGS.batch_size)
            # go through the batches
            for batch_num in range(total_batch_training):

                start_idx = batch_num * FLAGS.batch_size
                end_idx = (batch_num + 1) * FLAGS.batch_size

                # Fit training using batch data
                train_batch_data, train_batch_label = data['train/image'][start_idx:end_idx], data['train/label'][
                                                                                            start_idx:end_idx]
                # Run optimization op (backprop) and Calculate batch loss and accuracy
                # When the tensor tensors['global_step'] is evaluated, it will be incremented by one.
                batch_loss, _ = sess.run([loss, train_op],
                    feed_dict={image_place: train_batch_data,
                            label_place: train_batch_label,
                            dropout_param: 0.5})

            print("Epoch " + str(epoch + 1) + ", Training Loss= " + \
                "{:.5f}".format(batch_loss))

        # Evaluation of the model
        test_accuracy = 100 * sess.run(accuracy, feed_dict={
            image_place: data['test/image'],
            label_place: data['test/label'],
            dropout_param: 1.})
        print("Final Test Accuracy is %% %.2f" % test_accuracy)

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str,
                        default='/Users/yuchenwu/Desktop/TensorFlowModel/temp',
                        help='Directory for log data')
    parser.add_argument('--num_classes', type=int,
                        default=2,
                        help='Number of classes')
    parser.add_argument('--batch_size', type=int,
                        default=512,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int,
                        default=10,
                        help='Batch size')
    parser.add_argument('--initial_learning_rate', type=float,
                        default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        default=0.95,
                        help='X')
    parser.add_argument('--num_epochs_per_decay', type=int,
                        default=1,
                        help='X')
    parser.add_argument('--is_training', type=bool,
                        default=False,
                        help='X')
    parser.add_argument('--fine_tuning', type=bool,
                        default=False,
                        help='X')
    parser.add_argument('--online_test', type=bool,
                        default=True,
                        help='X')
    parser.add_argument('--allow_soft_placement', type=bool,
                        default=True,
                        help='X')
    parser.add_argument('--log_device_placement', type=bool,
                        default=False,
                        help='X')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)