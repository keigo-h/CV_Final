import tensorflow as tf
import random
import keras
import csv
import cv2
from matplotlib import pyplot as plt
import numpy as np
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional,Dropout
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

MAX_LEN = 0

class HandleData:
    def __init__(self, img_size) -> None:
        self.preprocess()
        self.img_size = img_size
        self.chars = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def preprocess(self,data_path="../words.txt"):
        chars = set()
        max_len = 0
        self.img_paths = []
        self.label_paths = []
        with open(data_path, 'r') as file:
            line = file.readline()
            while line:
                if line[0] != "#":
                    data_lst = line[:-1].split(" ")
                    if data_lst[1] == "ok":
                        path_lst = data_lst[0].split("-")
                        path = f"../words/{path_lst[0]}/{path_lst[0]}-{path_lst[1]}/{data_lst[0]}.png"
                        self.img_paths.append(path)
                        label = data_lst[8]
                        max_len = max(max_len, len(label))
                        self.label_paths.append(label)
                        for c in label:
                            chars.add(c)
                line = file.readline()
        
        assert len(self.img_paths) == len(self.label_paths)
        # temp = list(zip(self.img_paths, self.label_paths))
        # random.shuffle(temp)
        # self.img_paths, self.label_paths = zip(*temp)
        self.img_paths = self.img_paths[:5265]
        self.label_paths = self.label_paths[:5265]
        idx = int(len(self.img_paths) * 0.95)
        self.max_len = max_len
        self.train_img = self.img_paths[:idx]
        self.train_label = self.label_paths[:idx]
        temp = list(zip(self.train_img, self.train_label))
        random.shuffle(temp)
        self.train_img, self.train_label = zip(*temp)
        self.val_img = self.img_paths[idx:]
        self.val_label = self.label_paths[idx:]

    def encode_label(self, label):
        out = len(self.chars) * np.ones(self.max_len)
        for i,c in enumerate(label):
            out[i] = self.chars.find(c)
        return out
    
    def process_imgs(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros(self.img_size[::-1])
        
        w,h = img.shape
        aspect_width = 32
        aspect_height = int(h * (aspect_width / w))
        img = cv2.resize(img, (aspect_height, aspect_width))

        w, h = img.shape
        img = img.astype('float32')

        if w < 32:
            add_zeros = np.full((32-w, h), 255)
            img = np.concatenate((img, add_zeros))
            w, h = img.shape
    
        if h < 128:
            add_zeros = np.full((w, 128-h), 255)
            img = np.concatenate((img, add_zeros), axis=1)
            w, h = img.shape
            
        if h > 128 or w > 32:
            dim = (128,32)
            img = cv2.resize(img, dim)

        img = np.expand_dims(img, -1)
        # img = cv2.transpose(img)
        img = img / 255 - 0.5

        return img
    
    def process_train(self):
        print("process train")
        train_imgs = []
        train_labels = []
        train_labels_len = []
        train_input_len = []
        count = 0
        for path, label in zip(self.train_img, self.train_label):
            count+=1
            train_imgs.append(self.process_imgs(path))
            train_labels.append(self.encode_label(label))
            train_labels_len.append(len(label))
            train_input_len.append(32)
        return np.asarray(train_imgs), np.asarray(train_labels), np.asarray(train_labels_len), np.asarray(train_input_len)

    def process_val(self):
        print("process val")
        val_imgs = []
        val_labels = []
        val_labels_len = []
        val_input_len = []
        for path, label in zip(self.val_img, self.val_label):
            val_imgs.append(self.process_imgs(path))
            val_labels.append(self.encode_label(label))
            val_labels_len.append(len(label))
            val_input_len.append(32)
        return np.asarray(val_imgs), np.asarray(val_labels), np.asarray(val_labels_len), np.asarray(val_input_len)
    
