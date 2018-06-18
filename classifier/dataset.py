#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import random
import sys
import tensorflow as tf
import cv2
import os
from PIL import Image
from PIL import ImageFile
import numpy as np
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
folder_path = '/Volumes/Ryan/CheXNet/test_normal'
def get_lines(filename):
    lines = None
    with open(filename, "r") as f:
        lines = [line.strip() for line in f.readlines()[:100] if len(line.strip()) > 1]
    return np.array(lines)

def normalize(train_x, val_x, test_x):
    """normalize
    This function computes train mean and standard deviation on all pixels then applying data scaling on train_x, val_x and test_x using these computed values

    :param train_x: train samples, shape=(num_train, num_feature)
    :param val_x: validation samples, shape=(num_val, num_feature)
    :param test_x: test samples, shape=(num_test, num_feature)
    """
    # train_mean and train_std should have the shape of (1, 1)
    train_mean = np.mean(train_x, axis=(0,1), dtype=np.float64, keepdims=True)
    train_std = np.std(train_x, axis=(0,1), dtype=np.float64, keepdims=True)

    train_x = (train_x-train_mean)/train_std
    val_x = (val_x-train_mean)/train_std
    test_x = (test_x-train_mean)/train_std
    return train_x, val_x, test_x

class DataSet:
    def __init__(self):
        # one_data_file = '../0206/choose_normal_images.txt'
        # zero_data_file = '../0206/choose_bse_clahe_images.txt'
        # one_data = get_lines(one_data_file)
        # labels = np.ones((one_data.shape[0],))
        # zero_data = get_lines(zero_data_file)
        # labels = np.concatenate((labels, np.zeros((zero_data.shape[0],))), axis=0)
        # data = np.concatenate((one_data, zero_data), axis=0)
        # permutation = np.random.permutation(data.shape[0])
        # data = data[permutation]
        # labels = labels[permutation]
        #
        # limit = int(data.shape[0]*0.8)
        self.train_data = np.load("train_data.npy")
        self.train_labels = np.load("train_labels.npy")
        self.eval_data = np.load("eval_data.npy")
        self.eval_labels = np.load("eval_labels.npy")
        self.curr_training_step = 0
        self.curr_test_step = 0

    def get_train_set_size(self):
        return self.train_data.shape[0]

    def get_test_set_size(self):
        return self.eval_data.shape[0]

    def to_one_hot(self, X):
        one_hot = np.zeros((len(X), 2))
        for i in range(len(X)):
            np.put(one_hot[i, :], X[i], 1)

        return one_hot

    def get_image_data(self, batch_data):

        return np.array(batch)

    def next_batch(self, batch_size):
        X_train_bs = self.train_data[self.curr_training_step * batch_size:self.curr_training_step * batch_size + batch_size]
        Y_train_bs = self.train_labels[self.curr_training_step * batch_size:self.curr_training_step * batch_size + batch_size]

        self.curr_training_step = self.curr_training_step + 1
        self.curr_training_step = self.curr_training_step if (
            self.curr_training_step * batch_size < self.get_train_set_size()) else 0

        return (X_train_bs, self.to_one_hot(Y_train_bs))

    def next_batch_test(self, batch_size):
        # if self.eval_data is None:
        #     self.eval_data = np.load("train_data.npy")
        X_test_bs = self.eval_data[self.curr_test_step * batch_size:self.curr_test_step * batch_size + batch_size]
        Y_test_bs = self.eval_labels[self.curr_test_step * batch_size:self.curr_test_step * batch_size + batch_size]


        self.curr_test_step = self.curr_test_step + 1
        self.curr_test_step = self.curr_test_step if (self.curr_test_step * batch_size < self.get_test_set_size()) else 0

        return (X_test_bs, self.to_one_hot(Y_test_bs))

    def visualize_train_sample(self, idx):
        img = np.reshape(self.train_data[idx,:], [224,224])
        cv2.imshow('train sample', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
