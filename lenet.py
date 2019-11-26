import keras
import datetime
import numpy as np
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2, l1
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#from networks.train_plot import PlotLearning


# Code taken from https://github.com/BIGBALLON/cifar-10-cnn
class LeNet:
    def __init__(self, epochs=1000, batch_size=256, load_weights=True):
        self.name = 'lenet'
        self.model_filename = 'lenet.h5'
        self.num_classes = 8
        self.input_shape = 28, 28, 1
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_filepath = r'./lenet/lenet_tensorboard'

        if load_weights:
            try:
                self._model = load_model(self.model_filename)
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError):
                print('Failed to load', self.name)

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(6, (5, 5), strides=(1, 1), input_shape=(28, 28, 1), padding='valid',
                         kernel_initializer='uniform', ))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(16, (5, 5), strides=(1, 1), padding='valid', kernel_initializer='uniform', ))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Flatten())
        model.add(Dense(120, ))
        model.add(Dense(84, ))
        model.add(Dense(8, activation='sigmoid'))
        model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.00001),#0.00001
                      metrics=['accuracy'])
        return model

    def train(self):
        data_path = '../../0_AEEA_dataset/8class_of_traffic_dataset/data_norm8CLS_ALL.npy'
        label_path = '../../0_AEEA_dataset/8class_of_traffic_dataset/label_norm8CLS_ALL.npy'
        data = np.load(data_path)
        label_n = np.load(label_path)
        print(label_n[1:10])
        print('label的数量', label_n.shape)
        print("label的格式为：", type(label_n))
        data = data.reshape([-1, 28, 28, 1])
        # data = data*256
        x_train = data[:12417]
        y_train = label_n[:12417]

        x_test = data[12417:]
        y_test = label_n[12417:]

        x_test = x_test * 256
        x_test = x_test.astype(int)
        x_train = x_train * 256
        x_train = x_train.astype(int)
        y_test = y_test.astype(int)
        y_train = y_train.astype(int)
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # build network
        model = self.build_model()
        model.summary()

        # Save the best model during each training checkpoint
        checkpoint = ModelCheckpoint(self.model_filename,
                                     monitor='val_loss',
                                     verbose=0,
                                     save_best_only=True,
                                     mode='auto')
        tb_cb = TensorBoard(log_dir=self.log_filepath + '/' + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S"), histogram_freq=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, verbose=1,
                                      patience=20, min_lr=0.000000001)
        cbks = [checkpoint, tb_cb, ]

        model.fit(x_train, y_train, batch_size=self.batch_size,
                  verbose=2,
                  epochs=self.epochs,
                  callbacks=cbks,
                  validation_data=(x_test, y_test))
        # save model
        model.save(self.model_filename)

        self._model = model


if __name__ == '__main__':
    lenet_n = LeNet()
    lenet_n.train()
