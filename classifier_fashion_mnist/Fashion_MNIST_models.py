# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras import Model, regularizers
import os
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Fashion_MnistModel_one_layer(Model):
    def __init__(self):
        super(Fashion_MnistModel_one_layer, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(512, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        y = self.d1(x)
        return y


class Fashion_MnistModel_mullayer(Model):
    def __init__(self):
        super(Fashion_MnistModel_mullayer, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(256, activation='relu')
        self.d3 = Dense(128, activation='relu')
        self.d4 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        y = self.d4(x)
        return y


class Fashion_MnistModel_Dropout(Model):
    def __init__(self):
        super(Fashion_MnistModel_Dropout, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(256, activation='relu')
        self.d3 = Dense(128, activation='relu')
        self.drop = Dropout(0.5)
        self.d4 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.drop(x)
        y = self.d4(x)
        return y


class Fashion_MnistModel_BN(Model):
    def __init__(self):
        super(Fashion_MnistModel_BN, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.BN1 = BatchNormalization()
        self.d2 = Dense(256, activation='relu')
        self.BN2 = BatchNormalization()
        self.d3 = Dense(128, activation='relu')
        self.BN3 = BatchNormalization()
        self.d4 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.BN1(x)
        x = self.d2(x)
        x = self.BN2(x)
        x = self.d3(x)
        x = self.BN3(x)
        y = self.d4(x)
        return y


class Fashion_MnistModel_R(Model):
    def __init__(self):
        super(Fashion_MnistModel_R, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.d2 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.d3 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.d4 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        y = self.d4(x)
        return y


class Fashion_MnistModel_Optimize(Model):
    def __init__(self):
        super(Fashion_MnistModel_Optimize, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation='relu')  # , kernel_regularizer=regularizers.l2(0.005))
        self.BN1 = BatchNormalization()
        self.d2 = Dense(512, activation='relu')  # , kernel_regularizer=regularizers.l2(0.005))
        self.dropout1 = Dropout(0.2)
        self.BN2 = BatchNormalization()
        self.d3 = Dense(256, activation='relu')  # , kernel_regularizer=regularizers.l2(0.005))
        self.dropout2 = Dropout(0.2)
        self.BN3 = BatchNormalization()
        self.d4 = Dense(128, activation='relu')  # , kernel_regularizer=regularizers.l2(0.005))
        self.dropout3 = Dropout(0.5)
        self.d5 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.BN1(x)
        x = self.d2(x)
        # x = self.dropout1(x)
        x = self.BN2(x)
        x = self.d3(x)
        # x = self.dropout2(x)
        x = self.BN3(x)
        x = self.d4(x)
        x = self.dropout3(x)
        y = self.d5(x)
        return y


def generateds(path, txt):
    f = open(txt, 'r')
    contents = f.readlines()  # read in lines
    f.close()
    x, y_ = [], []
    for content in contents:
        value = content.split()  # split by Space, stored in array
        img_path = path + value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L'))
        img = img / 255.0
        x.append(img)
        y_.append(value[1])
    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)
    print("Generate Dateset successfully!")
    return x, y_


if __name__ == "__main__":
    np.set_printoptions(threshold=float('inf'))
    model_save_path = './checkpoint/fashion_mnist.ckpt'
    load_pretrain_model = False

    train_path = './fashion_mnist_image_label/fashion_mnist_train_jpg_60000/'
    train_txt = './fashion_mnist_image_label/fashion_mnist_train_jpg_60000.txt'
    test_path = './fashion_mnist_image_label/fashion_mnist_test_jpg_10000/'
    test_txt = './fashion_mnist_image_label/fashion_mnist_test_jpg_10000.txt'

    x_train, y_train = generateds(train_path, train_txt)
    x_test, y_test = generateds(test_path, test_txt)
    # 输出图片标签，展示图片
    # print('y_train[0]', y_train[0])
    # plt.imshow(x_train[0], cmap=plt.cm.binary)
    # plt.show()

    # model = Fashion_MnistModel_one_layer()
    # model = Fashion_MnistModel_mullayer()
    # model = Fashion_MnistModel_BN()
    # model = Fashion_MnistModel_Dropout()
    # model = Fashion_MnistModel_R()
    # model = Fashion_MnistModel_Optimize()
    _models = [Fashion_MnistModel_one_layer(),
               Fashion_MnistModel_mullayer(),
               Fashion_MnistModel_BN(),
               Fashion_MnistModel_Dropout(),
               Fashion_MnistModel_R()]

    for model in _models:
        model = model
        print("----------------Training on", model.name, "---------------------")
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01, decay=0.01),
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])

        log_dir = './logs_1/' + model.name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # 创建一个回调，为TensorBoard编写一个日志
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                     write_graph=True,
                                                     histogram_freq=1,
                                                     update_freq='epoch')

        model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test),
                  callbacks=[tb_callback], validation_freq=1, verbose=1)
        # model.summary()
