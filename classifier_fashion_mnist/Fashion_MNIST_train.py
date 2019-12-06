# -*- coding:utf-8 -*-

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=float('inf'))
model_save_path = './checkpoint/cp-00xx.ckpt'
load_pretrain_model = False

train_path = './fashion_mnist_image_label/fashion_mnist_train_jpg_60000/'
train_txt = './fashion_mnist_image_label/fashion_mnist_train_jpg_60000.txt'

test_path = './fashion_mnist_image_label/fashion_mnist_test_jpg_10000/'
test_txt = './fashion_mnist_image_label/fashion_mnist_test_jpg_10000.txt'


def generateds(path, txt):
    f = open(txt, 'r')
    contents = f.readlines()  # read in lines
    f.close()
    x, y_ = [], []
    for content in contents:
        value = content.split() # split by Space, stored in array
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


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


x_train, y_train = generateds(train_path, train_txt)
x_test, y_test = generateds(test_path, test_txt)

model = create_model()

if load_pretrain_model:
    print('-----------load the model-------------')
    model.load_weights(model_save_path)

model.summary

checkpoint_path = "./checkpoint/cp-{epoch:04d}.ckpt"
# 创建一个回调，每5个epochs保存模型的权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=5)

log_dir = './logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# 创建一个回调，为TensorBoard编写一个日志
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                             histogram_freq=1,
                                             update_freq='epoch')

# 使用新的回调训练模型
model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test),
          callbacks=[tb_callback, cp_callback], validation_freq=1, verbose=1)

# model.save_weights(model_save_path)

file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(v.name + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()
