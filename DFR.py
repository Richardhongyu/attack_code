# from __future__ import absolute_import, division, print_function, unicode_literals
# 导入TensorFlow和tf.keras
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
from tensorflow import keras
from keras import initializers
from keras import optimizers
from keras.callbacks import *
from keras.models import Sequential, load_model
from keras.layers import Conv2D, LSTM, Flatten, Dense, Activation, BatchNormalization, Dropout, Reshape, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l2
from keras.utils import multi_gpu_model
# from networks.train_plot import PlotLearning
# from __future__ import absolute_import, division, print_function, unicode_literals
# 导入TensorFlow和tf.keras
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from tensorflow import keras
from keras import initializers
from keras import optimizers
from keras.callbacks import *
from keras.models import Sequential, load_model
from keras.layers import Conv2D, LSTM, Flatten, Dense, Activation, BatchNormalization, Dropout, Reshape, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l2
from keras.utils import multi_gpu_model
# from networks.train_plot import PlotLearning

# 导入辅助库
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 检验tensorflow版本
print(tf.__version__)


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


class LR_Updater(Callback):
    '''This callback is utilized to log learning rates every iteration (batch cycle)
    it is not meant to be directly used as a callback but extended by other callbacks
    ie. LR_Cycle
    '''

    def __init__(self, iterations):
        '''
        iterations = dataset size / batch size
        epochs = pass through full training dataset
        '''

        self.epoch_iterations = iterations
        self.trn_iterations = 0.
        self.history = {}

    def on_train_begin(self, logs={}):
        self.trn_iterations = 0.
        logs = logs or {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        K.set_value(self.model.optimizer.lr, self.setRate())
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def plot_lr(self):
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(self.history['iterations'], self.history['lr'])

    def plot(self, n_skip=10):
        plt.xlabel("learning rate (log scale)")
        plt.ylabel("loss")
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')


class LR_Cycle(LR_Updater):
    '''This callback is utilized to implement cyclical learning rates
    it is based on this pytorch implementation https://github.com/fastai/fastai/blob/master/fastai
    and adopted from this keras implementation https://github.com/bckenstler/CLR
    '''

    def __init__(self, iterations, cycle_mult=1):
        '''
        iterations = dataset size / batch size
        iterations = number of iterations in one annealing cycle
        cycle_mult = used to increase the cycle length cycle_mult times after every cycle
        for example: cycle_mult = 2 doubles the length of the cycle at the end of each cy$
        '''
        self.min_lr = 0
        self.cycle_mult = cycle_mult
        self.cycle_iterations = 0.
        super().__init__(iterations)

    def setRate(self):
        self.cycle_iterations += 1
        if self.cycle_iterations == self.epoch_iterations:
            print(self.epoch_iterations, 'change')
            self.cycle_iterations = 0.
            self.epoch_iterations *= self.cycle_mult
        cos_out = np.cos(np.pi * (self.cycle_iterations) / self.epoch_iterations) + 1
        if (self.cycle_iterations % 10) == 0:
            print(self.max_lr / 2 * cos_out)
        # print(self.epoch_iterations)
        # print(np.pi * (self.cycle_iterations) / self.epoch_iterations)
        # print(self.cycle_iterations,'iters')
        # print(cos_out)
        # print(self.max_lr / 2 * cos_out,'begin')
        return self.max_lr / 2 * cos_out

    def on_train_begin(self, logs={}):
        super().on_train_begin(logs={})  # changed to {} to fix plots after going from 1 to mult. lr
        self.cycle_iterations = 0.
        self.max_lr = K.get_value(self.model.optimizer.lr)


# HAPPY
class DFR_model:
    def __init__(self, epochs=100000, batch_size=512, load_weights=True):
        self.name = 'DFR'
        self.model_filename = './DFR.h5'
        self.num_classes = 8
        self.input_shape = [28, 28, 1]
        self.epochs = epochs  #
        self.batch_size = batch_size  #
        self.weight_decay = 0.0001
        self.log_filepath = r'./DFR_tensorboard/'
        self.conv_l1_regularizer = 0.00045#
        # self.lstm_l1_regularizer = 0.0003  #
        self.start_lr = 0.001 #
        self.end_lr = 0.000001  #
        self.patience = 50  #
        self.epoch_1 = 1
        self.epoch_2 = 2
        self.epoch_3 = 3
        self.lr_1 = 0.001
        self.lr_2 = 0.001
        self.lr_3 = 0.001#0.55  0.5 0.475 0.04625 0.45 0.   4375 0.4

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
            Conv2D(32, (1, 25,), padding='SAME', strides=[1, 1, ],
                   kernel_initializer=initializers.random_normal(stddev=0.1),
                   kernel_regularizer=l1(self.conv_l1_regularizer)),
            # BatchNormalization(),
            # Dropout(0.5),
            Activation('relu'),
            MaxPooling2D((1, 3), strides=(1, 3), padding='SAME'),

            #
            # # CONV 2 Finished
            Conv2D(64, (1, 25,), padding='SAME', strides=[1, 1, ],
                   kernel_initializer=initializers.random_normal(stddev=0.1),
                   kernel_regularizer=l1(self.conv_l1_regularizer)),
            # BatchNormalization(),
            # Dropout(0.5),
            Activation('relu'),
            MaxPooling2D((1, 3), strides=(1, 3), padding='SAME'),

            #
            # # DENSE 1 / Dropout Finished
            Flatten(),
            Dense(1024, activation='relu',kernel_initializer=initializers.random_normal(stddev=0.1)),
            BatchNormalization(),
            Dropout(0.2),
            # Dropout(0.5),
            # DENSE 2 / SOFTMAX Finished
            # Dense(100, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01)),
            # Flatten(),
            Dense(8, activation='softmax', kernel_initializer=initializers.random_normal(stddev=0.1)),

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
        data_path = '/home/ailab/YI ZENG/Research/Classified/tra/DTrafficR/2_FusionedDataset/Numpy/data_norm8CLS_ALL.npy'
        label_path = '/home/ailab/YI ZENG/Research/Classified/tra/DTrafficR/2_FusionedDataset/Numpy/label_norm8CLS_ALL.npy'
        data = np.load(data_path)
        data = data.reshape([-1, 28, 28, 1])
        label_n = np.load(label_path)
        print('data的数量', data.shape)
        print('label的数量', label_n.shape)
        print(label_n[1:10])
        print("label的格式为：", type(label_n))

        x_train = data[:12417]
        y_train = label_n[:12417]
        x_test = data[12417:]
        y_test = label_n[12417:]

        # 数据归一化到【0：255】
        x_test = x_test * 256
        self.x_test = x_test.astype(int)
        x_train = x_train * 256
        self.x_train = x_train.astype(int)
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
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
                self.conv_l1_regularizer),
            histogram_freq=0)

        # lr change
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, verbose=1,
                                      patience=self.patience, min_lr=self.end_lr)

        # SGDR_lr = LR_Cycle(5000, 2)
        cbks = [checkpoint, tb_cb, reduce_lr]
        print('Using real-time data augmentation.')
        # datagen = ImageDataGenerator(horizontal_flip=False,
        #                              width_shift_range=0.01, height_shift_range=0.01, fill_mode='constant', cval=0.)
        #
        # datagen.fit(x_train)

        # start traing
        # model.fit_generator(datagen.flow(x_train, y_train,batch_size=self.batch_size),
        #                     steps_per_epoch=59,
        #                     verbose=2,
        #                     epochs=self.epochs,
        #                     callbacks=cbks,
        #                     validation_data=(x_test, y_test))
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
    DFR = DFR_model()
    DFR.train()
    print(DFR.accuracy())

    # best(val_acc:97): 0。003 0。001 0.000001/
    # goaled: 0.01 0.0005 0.0000001/0.003 0.003 0.000001/0。01 0。005 /0。003 0。001 0.000001/0.001 0.00013 0.0001
    # failed: 0.01 0.001 0.000001/

# 导入辅助库
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 检验tensorflow版本
print(tf.__version__)


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


class LR_Updater(Callback):
    '''This callback is utilized to log learning rates every iteration (batch cycle)
    it is not meant to be directly used as a callback but extended by other callbacks
    ie. LR_Cycle
    '''

    def __init__(self, iterations):
        '''
        iterations = dataset size / batch size
        epochs = pass through full training dataset
        '''

        self.epoch_iterations = iterations
        self.trn_iterations = 0.
        self.history = {}

    def on_train_begin(self, logs={}):
        self.trn_iterations = 0.
        logs = logs or {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        K.set_value(self.model.optimizer.lr, self.setRate())
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def plot_lr(self):
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(self.history['iterations'], self.history['lr'])

    def plot(self, n_skip=10):
        plt.xlabel("learning rate (log scale)")
        plt.ylabel("loss")
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')


class LR_Cycle(LR_Updater):
    '''This callback is utilized to implement cyclical learning rates
    it is based on this pytorch implementation https://github.com/fastai/fastai/blob/master/fastai
    and adopted from this keras implementation https://github.com/bckenstler/CLR
    '''

    def __init__(self, iterations, cycle_mult=1):
        '''
        iterations = dataset size / batch size
        iterations = number of iterations in one annealing cycle
        cycle_mult = used to increase the cycle length cycle_mult times after every cycle
        for example: cycle_mult = 2 doubles the length of the cycle at the end of each cy$
        '''
        self.min_lr = 0
        self.cycle_mult = cycle_mult
        self.cycle_iterations = 0.
        super().__init__(iterations)

    def setRate(self):
        self.cycle_iterations += 1
        if self.cycle_iterations == self.epoch_iterations:
            print(self.epoch_iterations, 'change')
            self.cycle_iterations = 0.
            self.epoch_iterations *= self.cycle_mult
        cos_out = np.cos(np.pi * (self.cycle_iterations) / self.epoch_iterations) + 1
        if (self.cycle_iterations % 10) == 0:
            print(self.max_lr / 2 * cos_out)
        # print(self.epoch_iterations)
        # print(np.pi * (self.cycle_iterations) / self.epoch_iterations)
        # print(self.cycle_iterations,'iters')
        # print(cos_out)
        # print(self.max_lr / 2 * cos_out,'begin')
        return self.max_lr / 2 * cos_out

    def on_train_begin(self, logs={}):
        super().on_train_begin(logs={})  # changed to {} to fix plots after going from 1 to mult. lr
        self.cycle_iterations = 0.
        self.max_lr = K.get_value(self.model.optimizer.lr)


# HAPPY
class DFR_model:
    def __init__(self, epochs=100000, batch_size=512, load_weights=True):
        self.name = 'DFR'
        self.model_filename = './DFR.h5'
        self.num_classes = 8
        self.input_shape = [28, 28, 1]
        self.epochs = epochs  #
        self.batch_size = batch_size  #
        self.weight_decay = 0.0001
        self.log_filepath = r'./DFR_tensorboard/'
        self.conv_l1_regularizer = 0.00045#
        # self.lstm_l1_regularizer = 0.0003  #
        self.start_lr = 0.001 #
        self.end_lr = 0.000001  #
        self.patience = 50  #
        self.epoch_1 = 1
        self.epoch_2 = 2
        self.epoch_3 = 3
        self.lr_1 = 0.001
        self.lr_2 = 0.001
        self.lr_3 = 0.001#0.55  0.5 0.475 0.04625 0.45 0.   4375 0.4

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
            Conv2D(32, (1, 25,), padding='SAME', strides=[1, 1, ],
                   kernel_initializer=initializers.random_normal(stddev=0.1),
                   kernel_regularizer=l1(self.conv_l1_regularizer)),
            # BatchNormalization(),
            # Dropout(0.5),
            Activation('relu'),
            MaxPooling2D((1, 3), strides=(1, 3), padding='SAME'),

            #
            # # CONV 2 Finished
            Conv2D(64, (1, 25,), padding='SAME', strides=[1, 1, ],
                   kernel_initializer=initializers.random_normal(stddev=0.1),
                   kernel_regularizer=l1(self.conv_l1_regularizer)),
            # BatchNormalization(),
            # Dropout(0.5),
            Activation('relu'),
            MaxPooling2D((1, 3), strides=(1, 3), padding='SAME'),

            #
            # # DENSE 1 / Dropout Finished
            Flatten(),
            Dense(1024, activation='relu',kernel_initializer=initializers.random_normal(stddev=0.1)),
            BatchNormalization(),
            Dropout(0.2),
            # Dropout(0.5),
            # DENSE 2 / SOFTMAX Finished
            # Dense(100, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01)),
            # Flatten(),
            Dense(8, activation='softmax', kernel_initializer=initializers.random_normal(stddev=0.1)),

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
        data_path = '/home/ailab/YI ZENG/Research/Classified/tra/DTrafficR/2_FusionedDataset/Numpy/data_norm8CLS_ALL.npy'
        label_path = '/home/ailab/YI ZENG/Research/Classified/tra/DTrafficR/2_FusionedDataset/Numpy/label_norm8CLS_ALL.npy'
        data = np.load(data_path)
        data = data.reshape([-1, 28, 28, 1])
        label_n = np.load(label_path)
        print('data的数量', data.shape)
        print('label的数量', label_n.shape)
        print(label_n[1:10])
        print("label的格式为：", type(label_n))

        x_train = data[:12417]
        y_train = label_n[:12417]
        x_test = data[12417:]
        y_test = label_n[12417:]

        # 数据归一化到【0：255】
        x_test = x_test * 256
        self.x_test = x_test.astype(int)
        x_train = x_train * 256
        self.x_train = x_train.astype(int)
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
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
                self.conv_l1_regularizer),
            histogram_freq=0)

        # lr change
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, verbose=1,
                                      patience=self.patience, min_lr=self.end_lr)

        # SGDR_lr = LR_Cycle(5000, 2)
        cbks = [checkpoint, tb_cb, reduce_lr]
        print('Using real-time data augmentation.')
        # datagen = ImageDataGenerator(horizontal_flip=False,
        #                              width_shift_range=0.01, height_shift_range=0.01, fill_mode='constant', cval=0.)
        #
        # datagen.fit(x_train)

        # start traing
        # model.fit_generator(datagen.flow(x_train, y_train,batch_size=self.batch_size),
        #                     steps_per_epoch=59,
        #                     verbose=2,
        #                     epochs=self.epochs,
        #                     callbacks=cbks,
        #                     validation_data=(x_test, y_test))
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
    DFR = DFR_model()
    DFR.train()
    print(DFR.accuracy())

    # best(val_acc:97): 0。003 0。001 0.000001/
    # goaled: 0.01 0.0005 0.0000001/0.003 0.003 0.000001/0。01 0。005 /0。003 0。001 0.000001/0.001 0.00013 0.0001
    # failed: 0.01 0.001 0.000001/
