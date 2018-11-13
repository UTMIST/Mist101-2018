"""
Linear Regression with random data.
To run this script:
    - python LinearRegresson.py
Check optional arguments:
    - python LinearRegression.py -h
"""
# System Import
import os
import argparse
import sys
# Tensorflow
import tensorflow as tf
# Other dependencies
import numpy as np
import matplotlib.pyplot as plt

# Used for argument parse
FLAGS = None

class LinearModel:

    def __init__(self):
        with tf.name_scope('Model/'):
            self.w = tf.Variable(0.0, name="Weights")
            self.b = tf.Variable(0.0, name="Bias")

    def forward(self, x):
        with tf.name_scope('Model/'):
            output = self.w * x + self.b
        return output

def compute_loss(prediction_y, label_y):
    with tf.name_scope('Loss'):
        loss = tf.reduce_sum(tf.squared_difference(label_y, prediction_y))/(2*FLAGS.num_data)
    return loss
    
def train(loss, learning_rate = 0.0001):
    with tf.name_scope('Gradient'):
        gradient = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return gradient

# Generating artificial data.
def generate_fake_data(n=50):
    """
    Forward passing the X.
    :param n: Number of fake data.
    :return: Fake data.
    """
    feature_x = np.arange(n)
    label_y = np.random.randint(-10, 10, size=50) + 2.0 * feature_x
    data = np.stack([feature_x,label_y], axis=1)
    return data

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
    # initial random seeds
    np.random.seed(0)
    # generate some random data
    data = generate_fake_data(FLAGS.num_data)
    # build graph
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, name="X")
        y = tf.placeholder(tf.float32, name="Y")
        model = LinearModel()
        # create graph: model (inference)
        prediction_y = model.forward(x)
        # create graph: loss function
        train_loss = compute_loss(prediction_y, y)
        # create graph: train
        train_op = train(train_loss, FLAGS.learning_rate)
    # training
    with tf.Session(graph=graph) as sess:
        # Initialize the variables[w and b].
        sess.run(tf.global_variables_initializer())
        # train the model
        for epoch_num in range(FLAGS.num_epoch):
            loss_value, _ = sess.run([train_loss,train_op],
                                    feed_dict={x: data[:,0], y: data[:,1]})
            # Displaying the loss per epoch.
            print('epoch %d, loss=%f' %(epoch_num+1, loss_value))
        # save the values of weight and bias TODO
        wcoeff = sess.run([model.w])
        bias = sess.run([model.b])
        try:
            writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            writer.close()
        except:
            print("Errorr: Change your directory!")
    # Plot the result.
    plot(data, wcoeff, bias)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str,
                        default='/home/yuchen/Desktop/Mist101-2018/TensorFlowModel',
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