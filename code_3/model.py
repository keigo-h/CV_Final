import tensorflow as tf
import random
import keras
import csv
import cv2
from matplotlib import pyplot as plt
import numpy as np
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional,Dropout, ReLU, Activation
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from preprocess import HandleData

char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def create_model():
    inputs = Input(shape=(32,128,1))

    model = Conv2D(32, (5,5), activation = 'relu', padding='same')(inputs)
    model = BatchNormalization()(model)
    model = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(model)

    model = Conv2D(64, (5,5), activation = 'relu', padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(model)

    model = Conv2D(128, (3,3), activation = 'relu', padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPool2D(pool_size=(1,2), strides=(1,2), padding='valid')(model)

    model = Conv2D(128, (3,3), activation = 'relu', padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPool2D(pool_size=(1,2), strides=(1,2), padding='valid')(model)

    model = Conv2D(256, (3,3), activation = 'relu', padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPool2D(pool_size=(1,2), strides=(1,2), padding='valid')(model)

    model = Reshape(target_shape=(32,256))(model)
    # model = Reshape(target_shape=(8,1024))(model)

    model = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(model)
    model = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(model)

    outputs = Dense(len(char_list)+1, activation = 'softmax')(model)

    the_model = Model(inputs, outputs)
    the_model.summary()
    return the_model,outputs,inputs

def create_model_3():
        # input with shape of height=32 and width=128 
    inputs = Input(shape=(32,128,1))

    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

    conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)

    conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)

    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(blstm_1)

    outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

    # model to be used at test time
    act_model = Model(inputs, outputs)
    act_model.summary()
    return act_model,outputs,inputs

def create_model_2():
    inputs = Input(shape=(32,128,1))
        ## Convolutional layers
    # layer 1 
    conv_1 = Conv2D(64, 3, strides=1, padding="same", kernel_initializer="he_normal", activation="relu", name="conv_1")(inputs)
    # layer 2
    conv_2 = Conv2D(32, 3, strides=1, padding="same", kernel_initializer="he_normal", activation="relu", name="conv_2")(conv_1)
    max_pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_2)
    # layer 3
    conv_3 = Conv2D(64, 3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal', name="conv_3")(max_pool_1)
    conv_4 = Conv2D(32, 3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal', name="conv_4")(conv_3)
    max_pool_2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_4)



    ### Encoding 
    reshape = Reshape(target_shape=((128//4), (32//4)*32), name="reshape_layer")(max_pool_2)
    dense_encoding = Dense(64, kernel_initializer="he_normal", activation="relu", name="enconding_dense")(reshape)
    dense_encoding_2 = Dense(64, kernel_initializer="he_normal", activation="relu", name="enconding_dense_2")(dense_encoding)
    dropout = Dropout(0.4)(dense_encoding_2)

    # Decoder
    lstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25), name="bidirectional_lstm_1")(dropout)
    lstm_2 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.25), name="bidirectional_lstm_2")(lstm_1)
    outputs = Dense(len(char_list)+1, activation = 'softmax')(lstm_2)

    act_model = Model(inputs, outputs)
    act_model.summary()
    return act_model,outputs,inputs


def ctc_loss_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def gen_model(hD: HandleData):
    train_img, train_label, train_label_len, train_inp_len = hD.process_train()
    val_img, val_label, val_label_len, val_inp_len = hD.process_val()
    test_model, out, inp = create_model()
    the_labels = Input(shape=[hD.max_len], dtype='float32')
    input_length = Input(shape=[1], dtype='int64')
    label_length = Input(shape=[1], dtype='int64')
    loss_out = Lambda(ctc_loss_func, output_shape=(1,), name='ctc')([out, the_labels, input_length, label_length])

    model = Model(inputs=[inp, the_labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'adam', metrics=['accuracy'])

    filepath = "cur_best.hdf5"

    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    history = model.fit(x=[train_img, train_label, train_inp_len, train_label_len],
                    y=np.zeros(len(train_img)),
                    batch_size=100, 
                    epochs=20, 
                    validation_data=([val_img, val_label, val_inp_len, val_label_len], [np.zeros(len(val_img))]),
                    verbose=1,
                    callbacks=callbacks_list)
    # model.save(filepath='./model3.h5', overwrite=False, include_optimizer=True)
    model.save(filepath='./model.h5', overwrite=False, include_optimizer=True)

def main():
    # create_model()
    hD = HandleData(img_size=(32,128))
    print(hD.max_len)
    gen_model(hD)

main()