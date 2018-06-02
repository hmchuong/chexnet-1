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

ImageFile.LOAD_TRUNCATED_IMAGES = True
folder_path = '/Volumes/Ryan/CheXNet/test_normal'
def get_lines(filename):
    lines = None
    with open(filename, "r") as f:
        lines = [line.strip() for line in f.readlines() if len(line.strip()) > 1]
    return np.array(lines)

class DataSet:
    def __init__(self):
        one_data_file = '../0206/choose_normal_images.txt'
        zero_data_file = '../0206/choose_bse_clahe_images.txt'
        one_data = get_lines(one_data_file)
        labels = np.ones((one_data.shape[0],))
        zero_data = get_lines(zero_data_file)
        labels = np.concatenate((labels, np.zeros((zero_data.shape[0],))), axis=0)
        data = np.concatenate((one_data, zero_data), axis=0)
        permutation = np.random.permutation(data.shape[0])
        data = data[permutation]
        labels = labels[permutation]

        limit = int(data.shape[0]*0.8)
        self.train_data = data[:limit]  # Returns np.array
        self.train_labels = labels[:limit]
        self.eval_data = data[limit:]
        self.eval_labels = labels[limit:]
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
        batch = []
        for path in batch_data:
            ds = Image.open(os.path.join(folder_path, path)).convert("L")
            ds = ds.resize((224, 224))
            #print(ds.size)
            arr = np.array(ds).reshape((-1,))
            #if (arr.shape[0] > 50176) :
                #print(path)
            #print("ARR", arr.shape)
            batch.append(arr)
            # try:
            #
            #     try:
            #
            #     except Exception as e:
            #         print("Cannot resize image {} with size {}".format(path, ds))
            #         print(e)
            # except Exception as e:
            #     print("Cannot read image {}".format(path))
            #     print(e)
        #print ("BATCH ", len(batch), len(batch_data))
        return np.array(batch)

    def next_batch(self, batch_size):
        X_train_bs = self.get_image_data(self.train_data[self.curr_training_step * batch_size:self.curr_training_step * batch_size + batch_size])
        Y_train_bs = self.train_labels[self.curr_training_step * batch_size:self.curr_training_step * batch_size + batch_size]

        self.curr_training_step = self.curr_training_step + 1
        self.curr_training_step = self.curr_training_step if (
            self.curr_training_step * batch_size < self.get_train_set_size()) else 0

        return (X_train_bs, self.to_one_hot(Y_train_bs))

    def next_batch_test(self, batch_size):
        X_test_bs = self.get_image_data(self.eval_data[self.curr_test_step * batch_size:self.curr_test_step * batch_size + batch_size])
        Y_test_bs = self.eval_labels[self.curr_test_step * batch_size:self.curr_test_step * batch_size + batch_size]


        self.curr_test_step = self.curr_test_step + 1
        self.curr_test_step = self.curr_test_step if (self.curr_test_step * batch_size < self.get_test_set_size()) else 0

        return (X_test_bs, self.to_one_hot(Y_test_bs))

    def visualize_train_sample(self, idx):
        img = np.reshape(self.train_data[idx,:], [224,224])
        cv2.imshow('train sample', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
