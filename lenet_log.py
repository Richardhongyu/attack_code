import keras
import datetime
import numpy as np
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2,l1
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#from networks.train_plot import PlotLearning



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
# Code taken from https://github.com/BIGBALLON/cifar-10-cnn
class LeNet1:
    def __init__(self, epochs=10000, batch_size=512, load_weights=True):
        self.name = 'lenet_log'
        self.model_filename = './lenet_log.h5'
        self.num_classes = 2
        self.input_shape = 28, 28, 1
        self.batch_size = batch_size
        self.epochs = epochs
        self.iterations = 400
        self.weight_decay = 0.00005
        self.log_filepath = r'./lenet_log_tensorboard/'

        if load_weights:
            try:
                self._model = load_model(self.model_filename)
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError):
                print('Failed to load', self.name)

    def count_params(self):
        return self._model.count_params()

    def color_preprocessing(self, x_train, x_test):
        x_train = x_train.astype('float64')
        x_test = x_test.astype('float64')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
            x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
        return x_train, x_test

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(6, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.weight_decay), input_shape=self.input_shape))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(16, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.weight_decay)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(
            Dense(120, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay)))
        model.add(
            Dense(84, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay)))
        model.add(
            Dense(2, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay)))
        sgd = optimizers.SGD(lr=0.0000095,)#0.000001
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def scheduler(self, epoch):
        print(epoch, '--------------------------')
        if epoch <= 20:
            return 0.0004
        if epoch <= 30:
            return 0.00008
        if epoch <= 50:
            return 0.00002
        return 0.000004

    def train(self):
        # data_path = '/home/ailab/YI ZENG/Research/Classified/tra/DTrafficR/2_FusionedDataset/Numpy/data_norm8CLS_ALL.npy'
        # label_path = '/home/ailab/YI ZENG/Research/Classified/tra/DTrafficR/2_FusionedDataset/Numpy/label_norm8CLS_ALL.npy'
        # data = np.load(data_path)
        # label_n = np.load(label_path)
        # print(label_n[1:10])
        # print('label的数量', label_n.shape)
        # print("label的格式为：", type(label_n))
        # data = data.reshape([-1, 28, 28, 1])
        # # data = data*256
        # x_train = data[:12417]
        # y_train = label_n[:12417]
        #
        # x_test = data[12417:]
        # y_test = label_n[12417:]
        #
        # x_test = x_test * 256
        # x_test = x_test.astype(int)
        # x_train = x_train * 256
        # x_train = x_train.astype(int)
        # y_test = y_test.astype(int)
        # y_train = y_train.astype(int)
        #
        # # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # y_train = keras.utils.to_categorical(y_train, self.num_classes)
        # y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # color preprocessing
        # x_train, x_test = self.color_preprocessing(x_train, x_test)

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

        # build network
        model = self.build_model()
        model.summary()

        mkdir(self.model_filename + 'date_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        # Save the best model during each training checkpoint
        change_lr = LearningRateScheduler(self.scheduler)
        checkpoint = ModelCheckpoint(self.model_filename + 'date_' + datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S") + '/' + 'epoch_' + '{epoch:02d}' + '_val_acc_' + '{val_acc:.4f}' + '.h5',
                                     monitor='val_acc',
                                     verbose=0,
                                     save_best_only=True,
                                     mode='auto')
        plot_callback = PlotLearning()
        tb_cb = TensorBoard(log_dir=self.log_filepath + '/' + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S"), histogram_freq=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, verbose=1,
                                      patience=20, min_lr=0.000000001)
        # tb_cb = TensorBoard(
        #     log_dir=self.log_filepath + 'date_' + datetime.datetime.now().strftime(
        #         "%Y%m%d-%H%M%S") + '_conv_l1_' + str(self.conv_l1_regularizer) + '_lstm_l1_' + str(
        #         self.lstm_l1_regularizer),
        #     histogram_freq=0)

        cbks = [checkpoint, tb_cb, ]

        # using real-time data augmentation
        print('Using real-time data augmentation.')
        # datagen = ImageDataGenerator(horizontal_flip=True,
        #                              width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant', cval=0.)
        #
        # datagen.fit(x_train)

        # start traing
        # model.fit_generator(datagen.flow(x_train, y_train,batch_size=self.batch_size),
        #                     steps_per_epoch=self.iterations,
        #                     verbose=2,
        #                     epochs=self.epochs,
        #                     callbacks=cbks,
        #                     validation_data=(x_test, y_test))
        model.fit(x=self.x_train, y=self.y_train, batch_size=self.batch_size,
                  verbose=2,
                  epochs=self.epochs,
                  callbacks=cbks,
                  validation_data=(self.x_test, self.y_test))
        # save model
        model.save(self.model_filename + '.h5')

        self._model = model

    def color_process(self, imgs):
        if imgs.ndim < 4:
            imgs = np.array([imgs])
        imgs = imgs.astype('float64')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for img in imgs:
            for i in range(1):
                img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
        return imgs

    def predict(self, img):
        # processed = self.color_process(img)
        img = img.astype('float64')
        processed = img
        # model = load_model('lenet.h5')
        # result_test=model.predict(processed)
        # print(result_test,"please sucess!!!!!!!!!!!!!!!!!!")
        # return result_test
        # print(model)
        # print(self._model.predict(processed, batch_size=self.batch_size))
        # print('processed shape is:')
        # print(processed.shape)
        # print('test 1', processed.shape)
        # print('test 2', processed.shape)
        return self._model.predict(processed, batch_size=self.batch_size)

    def predict_one(self, img):
        # print('g-------------------------------------------------------')
        return self.predict(img)[0]

    def accuracy(self):
        # data_path = '/home/ailab/YI ZENG/Research/Classified/tra/DTrafficR/2_FusionedDataset/Numpy/data_norm8CLS_ALL.npy'
        # label_path = '/home/ailab/YI ZENG/Research/Classified/tra/DTrafficR/2_FusionedDataset/Numpy/label_norm8CLS_ALL.npy'
        # data = np.load(data_path)
        # label_n = np.load(label_path)
        # print(label_n[1:10])
        # print('label的数量', label_n.shape)
        # print("label的格式为：", type(label_n))
        # data = data.reshape([-1, 28, 28, 1])
        # x_train = data[:12417]
        # y_train = label_n[:12417]
        #
        # x_test = data[12417:]
        # y_test = label_n[12417:]
        #
        # # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # y_train = keras.utils.to_categorical(y_train, self.num_classes)
        # y_test = keras.utils.to_categorical(y_test, self.num_classes)
        #
        # # color preprocessing
        # # x_train, x_test = self.color_preprocessing(x_train, x_test)

        return self._model.evaluate(self.x_test, self.y_test, verbose=0)[1]


if __name__ == '__main__':
    lenet = LeNet1()
    lenet.train()
    print(lenet.accuracy())
