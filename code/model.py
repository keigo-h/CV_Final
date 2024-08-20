import numpy as np
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
import keras.backend
from keras.callbacks import ModelCheckpoint, CSVLogger
from preprocess import HandleData
import params as p

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

    model = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(model)
    model = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(model)

    outputs = Dense(len(p.char_lst)+1, activation = 'softmax')(model)

    the_model = Model(inputs, outputs)
    return the_model,outputs,inputs

def ctc_loss_func(args):
    y_pred, labels, input_length, label_length = args
    return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def gen_model(hD: HandleData):
    train_img, train_label, train_label_len, train_inp_len = hD.process_train()
    val_img, val_label, val_label_len, val_inp_len = hD.process_val()
    the_model, out, inp = create_model()
    the_model.summary()
    the_labels = Input(shape=[hD.max_len], dtype='float32')
    input_length = Input(shape=[1], dtype='int64')
    label_length = Input(shape=[1], dtype='int64')
    loss_out = Lambda(ctc_loss_func, output_shape=(1,), name='ctc')([out, the_labels, input_length, label_length])

    model = Model(inputs=[inp, the_labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'adam', metrics=['accuracy'])

    val_loss_filepath = "val_loss_cur_best.hdf5"
    train_loss_filepath = "train_loss_cur_best.hdf5"

    checkpoint1 = ModelCheckpoint(filepath=val_loss_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    checkpoint2 = ModelCheckpoint(filepath=train_loss_filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    checkpoint3 = CSVLogger('res.csv', append=True, separator=';')
    callbacks_list = [checkpoint1, checkpoint2, checkpoint3]

    history = model.fit(x=[train_img, train_label, train_inp_len, train_label_len],
                    y=np.zeros(len(train_img)),
                    batch_size=p.batch_size, 
                    epochs=p.epochs, 
                    validation_data=([val_img, val_label, val_inp_len, val_label_len], [np.zeros(len(val_img))]),
                    verbose=1,
                    callbacks=callbacks_list)

    model.save(filepath='./model.h5', overwrite=True, include_optimizer=True)

def main():
    hD = HandleData(img_size=p.img_size)
    gen_model(hD)

if __name__ == "__main__":
    main()