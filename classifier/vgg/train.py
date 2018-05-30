from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2
from dataset import DataSet
import time
import pdb
import os
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg19:
    def __init__(self, sess=None, vgg19_npy_path=None, log=True, dropout=0.5, trainable=True):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout
        self.log = log
        self.sess = sess
        self.X = tf.placeholder(tf.float32, [None, 224, 224, 3], name='X')
        self.build(self.X)
        train_mode = tf.placeholder(tf.bool)

        print(self.get_var_count())
        print("npy file loaded")

    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, 2, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

    def train(self, learning_rate, training_epochs=40, batch_size=1000):
        # Load dataset for training and testing
        self.dataset = DataSet()

        self.Y = tf.placeholder(tf.float32, [None, 2], name='Y')

        # Define cost function
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prob, labels=self.Y))#(logits=self.prob, labels=self.Y))#tf.reduce_sum((self.prob - self.Y) ** 2)
        # Define optimization method
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Start logger
        if self.log:
            tf.summary.scalar('cost', self.cost)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('./log_train', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        print('Training...')
        weights = []
        # For each epoch, feed training data and perform updating parameters
        for epoch in range(training_epochs):
            avg_cost = 0
            # Number of batches = size of training set / batch_size
            total_batch = int(self.dataset.get_train_set_size() / batch_size)

            # For each batch
            for i in range(total_batch + 1):
                # Get next batch to feed to the network
                batch_xs, batch_ys = self.dataset.next_batch(batch_size)
                feed_dict = {
                    self.X: batch_xs.reshape([batch_xs.shape[0], 224, 224, 3]),
                    self.Y: batch_ys
                }

                summary, c, _ = self.sess.run([self.merged, self.cost, self.optimizer],
                                                       feed_dict=feed_dict)
                avg_cost += c / total_batch
                print('Step:', '%02d/' % (i + 1), '%02d' % (total_batch), 'cost =', '{:.9f}'.format(c))

            if self.log:
                self.train_writer.add_summary(summary, epoch + 1)

            print('Epoch:', '%02d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

        print('Training finished!')

        self.save_npy(sess, './worst_best_model.npy')

    def evaluate(self, batch_size):

      self.correct_prediction = tf.equal(tf.argmax(self.prob, 1), tf.argmax(self.Y, 1))
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

      N = self.dataset.get_test_set_size()
      print('test.size', N);
      correct_sample = 0
      for i in range(0, N, batch_size):
          batch_xs, batch_ys = self.dataset.next_batch_test(batch_size)

          N_batch = batch_xs.shape[0]

          feed_dict = {
              self.X: batch_xs.reshape([N_batch, 224, 224, 3]),
              self.Y: batch_ys
          }

          correct = self.sess.run(self.accuracy, feed_dict=feed_dict)
          correct_sample += correct * N_batch

      test_accuracy = correct_sample / N

      print("\nAccuracy Evaluates")
      print("-" * 30)
      print('Test Accuracy:', test_accuracy)

def main(unused_argv):
  sess = tf.Session()
  vgg = Vgg19(sess=sess)
  start = time.time()
  vgg.train(learning_rate=0.001, training_epochs=40, batch_size=16)
  vgg.evaluate(batch_size=16)
  end = time.time()
  print("Total time: {} seconds".format(end - start))

if __name__ == "__main__":
  tf.app.run()
