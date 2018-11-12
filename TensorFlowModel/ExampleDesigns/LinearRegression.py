"""
Linear Regression with random data
"""

import os
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

# Used for generating random data.
from sklearn.utils import check_random_state

# used for argument parse
FLAGS = None

# Generating artificial data.
def generate_fake_data(n=50):
    """
    Forward passing the X.
    :param n: Number of fake data.
    :return: Fake data.
    """
    XX = np.arange(n)
    rs = check_random_state(0)
    YY = rs.randint(-10, 10, size=(n,)) + 2.0 * XX
    data = np.stack([XX,YY], axis=1)
    return data

def linear_model(X):
    """
    Forward passing the X.
    :param X: Input.
    :return: X*W + b.
    """
    # creating the weight and bias.
    # The defined variables will be initialized to zero.
    with tf.name_scope('model'):
        W = tf.Variable(0.0, name="weights")
        b = tf.Variable(0.0, name="bias")
        output = X * W + b
    return output

def square_loss(Y_predict, Y, data):
    """
    compute the loss by comparing the predicted value to the actual label.
    :param X: The output from model.
    :param Y: The label.
    :return: The loss over the samples.
    """
    with tf.name_scope('loss'):
        loss = tf.reduce_sum(tf.squared_difference(Y, Y_predict))/(2*data.shape[0])
    return loss

# The training function.
def train(loss, learning_rate = 0.0001):
    with tf.name_scope('gradient'):
        gradient = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return gradient


def plot(data, wcoeff, bias):
    input_values = data[:,0]
    labels = data[:,1]
    prediction_values = data[:,0] * wcoeff + bias
    # uncomment if plotting is desired!
    plt.plot(input_values, labels, 'ro', label='main')
    plt.plot(input_values, prediction_values, label='Predicted')
    # Saving the result.
    plt.legend()
    plt.show()
    plt.close()

def main(_):
    # generate some random data
    data = generate_fake_data(FLAGS.num_data)
    # build graph
    graph = tf.Graph()
    with graph.as_default():
        # input data
        X = tf.placeholder(tf.float32, name="X")
        Y = tf.placeholder(tf.float32, name="Y")
        # create graph: model (inference)
        Y_predict = linear_model(X)
        # create graph: loss function
        train_loss = square_loss(Y_predict, Y, data)
        # create graph: train
        train_op = train(train_loss, FLAGS.learning_rate)
    # training
    with tf.Session(graph=graph) as sess:
        # Initialize the variables[w and b].
        sess.run(tf.global_variables_initializer())
        # train the model
        for epoch_num in range(FLAGS.num_epoch):
            loss_value, _ = sess.run([train_loss,train_op],
                                    feed_dict={X: data[:,0], Y: data[:,1]})
            # Displaying the loss per epoch.
            print('epoch %d, loss=%f' %(epoch_num+1, loss_value))
        # save the values of weight and bias TODO
        # wcoeff, bias = sess.run([W, b])

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        writer.close()
    # plot the result
    # plot(data, wcoeff, bias)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str,
                        default='/Users/yuchenwu/Desktop/TensorFlowModel/temp',
                        help='Directory for log data')
    parser.add_argument('--num_epoch', type=int,
                        default=50,
                        help='Number of epochs')
    parser.add_argument('--num_data', type=int,
                        default=50,
                        help='Number of fake data')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001,
                        help='Number of fake data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)