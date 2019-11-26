import tensorflow as tf
from tensorflow import keras
from keras import initializers
from keras import optimizers
from keras.callbacks import *
from keras.models import Sequential, load_model
from keras.layers import Conv2D, LSTM, Flatten, Dense, BatchNormalization, Dropout, Reshape, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l2
from keras.utils import multi_gpu_model

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import tensorflow as tf
from tensorflow import keras


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 检验tensorflow版本
print(tf.__version__)


# strategy = tf.distribute.MirroredS    trategy(devices=["/device:GPU:0", "/device:GPU:1"],
#                                           cross_device_ops=tf.contrib.distribute.AllReduceCrossDeviceOps(
#                                               all_reduce_alg="hierarchical_copy")
#                                           )
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


# HAPPY
class happy_model1:
    def __init__(self, epochs=10000, batch_size=2048, load_weights=True):
        self.name = 'happy'
        self.model_filename = './happy_log.h5'
        self.num_classes = 2
        self.input_shape = [28, 28, 1]
        self.epochs = epochs#
        self.batch_size = batch_size#
        self.weight_decay = 0.0001
        self.log_filepath = r'./happy_log_tensorboard/'
        # self.log_filepath = r'./happy_tensorboard/'
        self.conv_l1_regularizer = 0.003#
        self.lstm_l1_regularizer = 0.0011#
        self.start_lr = 0.001#
        self.end_lr = 0.000001#
        self.patience = 50#
        self.epoch_1 = 1
        self.epoch_2 = 2
        self.epoch_3 = 3
        self.lr_1 = 0.001
        self.lr_2 = 0.001
        self.lr_3 = 0.001

        if load_weights:
            try:
                self._model = load_model(self.model_filename)
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError) as e:
                print(e)
                print('Failed to load', self.name)

    def count_params(self):
        return self._model.count_params()

    def build_model(self):
        # self.batch_size = self.batch_size * strategy.num_replicas_in_sync
        # with strategy.scope():
        model = Sequential([

            # # FLATTEN Finishedsparse_
            Reshape((-1, 784, 1), input_shape=self.input_shape),
            #
            # # CONV 1 Finished
            Conv2D(32, (1, 25,), padding='SAME', strides=[1, 1, ], activation='relu',
                   kernel_initializer=initializers.random_normal(stddev=0.1),
                   kernel_regularizer=l1(self.conv_l1_regularizer)),
            BatchNormalization(),
            MaxPooling2D((1, 3), strides=(1, 3), padding='SAME'),
            #
            # # CONV 2 Finished
            Conv2D(64, (1, 25,), padding='SAME', strides=[1, 1, ], activation='relu',
                   kernel_initializer=initializers.random_normal(stddev=0.1),
                   kernel_regularizer=l1(self.lstm_l1_regularizer)),
            BatchNormalization(),
            MaxPooling2D((1, 3), strides=(1, 3), padding='SAME'),
            #
            # # DENSE 1 / Dropout Finished
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Reshape((32, 32)),
            #
            # # LSTM 1-3 Finished
            LSTM(256, dropout=0.5, return_sequences=True, kernel_regularizer=l1(self.lstm_l1_regularizer)),
            LSTM(256, dropout=0.5, return_sequences=True, kernel_regularizer=l1(self.lstm_l1_regularizer)),
            LSTM(256, dropout=0.5, return_sequences=False, kernel_regularizer=l1(self.lstm_l1_regularizer)),

            # DENSE 2 / SOFTMAX Finished
            # Dense(100, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01)),
            # Flatten(),
            Dense(2, activation='softmax', kernel_initializer=initializers.random_normal(stddev=0.01)),

        ])
        adam = optimizers.Adam(lr=self.start_lr, beta_1=0.9, beta_2=0.999, )  # 7.28增大10 times训练步长
        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # sparse_
        return model

    def scheduler(self, epoch):
        # print(epoch, '--------------------------')
        if epoch <= self.epoch_1:
            return self.lr_1
        if epoch <= self.epoch_2:
            return self.lr_2
        if epoch <= self.epoch_3:
            return self.lr_3
        return self.lr_3

    def train(self):
        # data_path = '/home/ailab/YI ZENG/Research/Classified/tra/DTrafficR/2_FusionedDataset/Numpy/data_norm8CLS_ALL.npy'
        # label_path = '/home/ailab/YI ZENG/Research/Classified/tra/DTrafficR/2_FusionedDataset/Numpy/label_norm8CLS_ALL.npy'
        # data = np.load(data_path)
        # data = data.reshape([-1, 28, 28, 1])
        # label_n = np.load(label_path)
        # print('data的数量', data.shape)
        # print('label的数量', label_n.shape)
        # print(label_n[1:10])
        # print("label的格式为：", type(label_n))
        #
        # x_train = data[:12417]
        # y_train = label_n[:12417]
        # x_test = data[12417:]
        # y_test = label_n[12417:]
        #
        # # 数据归一化到【0：255】
        # x_test = x_test * 256
        # self.x_test = x_test.astype(int)
        # x_train = x_train * 256
        # self.x_train = x_train.astype(int)
        # y_train = keras.utils.to_categorical(y_train, self.num_classes)
        # y_test = keras.utils.to_categorical(y_test, self.num_classes)
        # self.y_test = y_test.astype(int)
        # self.y_train = y_train.astype(int)

        train_data_path = './data_dfr_log/train_data.npy'
        train_label_path = './data_dfr_log/train_label.npy'
        test_data_path = './data_dfr_log/test_data.npy'
        test_label_path = './data_dfr_log/test_label.npy'

        train_data = np.load(train_data_path)
        train_label = np.load(train_label_path)
        test_data = np.load(test_data_path)
        test_label = np.load(test_label_path)

        print('train_data的数量为:', train_data.shape)
        print('train_label的数量为:', train_label.shape)
        print('test_data的数量为:', test_data.shape)
        print('test_label的数量为:', test_label.shape)

        train_data = train_data.reshape([-1, 28, 28, 1])
        # train_label = train_label.reshape([-1, 28, 28, 1])
        test_data = test_data.reshape([-1, 28, 28, 1])
        # test_label = test_label.reshape([-1, 28, 28, 1])

        print('train_data的数量为:', train_data.shape)
        print('train_label的数量为:', train_label.shape)
        print('test_data的数量为:', test_data.shape)
        print('test_label的数量为:', test_label.shape)
        # 数据归一化到【0：255】

        self.x_test = test_data.astype(int)
        self.x_train = train_data.astype(int)
        self.x_test = 2 * self.x_test
        self.x_train = 2 * self.x_train
        y_train = keras.utils.to_categorical(train_label, self.num_classes)
        y_test = keras.utils.to_categorical(test_label, self.num_classes)
        self.y_test = y_test.astype(int)
        self.y_train = y_train.astype(int)

        # 模型
        model = self.build_model()
        model.summary()

        # 参数文件夹保存
        mkdir(self.model_filename + 'date_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        # 训练
        change_lr = LearningRateScheduler(self.scheduler)

        checkpoint = ModelCheckpoint(
            self.model_filename + 'date_' + datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S") + '/' + 'epoch_' + '{epoch:02d}' + '_val_acc_' + '{val_acc:.4f}' + '.h5',
            monitor='val_acc',
            verbose=0,
            save_best_only=True,
            mode='auto',
            period=5)
        # plot_callback = PlotLearning()
        tb_cb = TensorBoard(
            log_dir=self.log_filepath + 'date_' + datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S") + '_conv_l1_' + str(self.conv_l1_regularizer) + '_lstm_l1_' + str(
                self.lstm_l1_regularizer),
            histogram_freq=0)

        # lr change


        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, verbose=1,
                                      patience=self.patience, min_lr=self.end_lr)

        #SGDR_lr = LR_Cycle(4000, 1.5)
        cbks = [checkpoint, tb_cb, reduce_lr]


        # start traing
        model.fit(x=self.x_train, y=self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  callbacks=cbks,
                  verbose=2,
                  validation_data=(self.x_test, self.y_test),
                  )
        # save model
        model.save(self.model_filename + '.h5')

        self._model = model

    def predict(self, img):
        return self._model.predict(img, batch_size=self.batch_size)

    def predict_one(self, img):
        return self.predict(img)[0]

    def accuracy(self):
        return self._model.evaluate(self.x_test, self.y_test, verbose=0)[1]


if __name__ == '__main__':
    happy = happy_model1()
    happy.train()
    print(happy.accuracy())

    # best(val_acc:97): 0。003 0。001 0.000001/
    # goaled: 0.01 0.0005 0.0000001/0.003 0.003 0.000001/0。01 0。005 /0。003 0。001 0.000001/0.001 0.00013 0.0001
    # failed: 0.01 0.001 0.000001/
