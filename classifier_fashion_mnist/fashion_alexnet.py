# !/usr/bin/env python
# coding:utf-8
# Author: Yuanjie
# 利用class结构训练并测试fashion_mnist
import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 给数据增加一个维度，使数据和网络结构匹配
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train, x_test = x_train / 255.0, x_test / 255.0  # 神经网络只对微小的振动敏感

image_gen_train = ImageDataGenerator(
                                     rescale=1,#不归至0～1  # 已经在上面对train和test进行过归一化了
                                     rotation_range=0,#随机0度旋转
                                     width_shift_range=0,#宽度偏移
                                     height_shift_range=0,#高度偏移
                                     horizontal_flip=True,#水平翻转
                                     zoom_range=1#将图像随机缩放到100％
                                     )
image_gen_train.fit(x_train)

class Fashion_Model(Model):
    def __init__(self):
        super(Fashion_Model, self).__init__()
        self.c1 = Conv2D(input_shape=(28, 28, 1), filters=96, kernel_size=(3, 3), padding='valid')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=2)  # 池化层
        self.c2 = Conv2D(filters=256, kernel_size=(3, 3), padding='valid')
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=2)
        self.c3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')  # 卷积层
        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same', activation='relu')  # 卷积层
        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')  # 卷积层
        self.p3 = MaxPool2D(pool_size=(3, 3), strides=2)
        self.flatten = Flatten()
        self.f1 = Dense(2048, activation='relu')
        self.d1 = Dropout(0.5)  # dropout层
        self.f2 = Dense(2048, activation='relu')
        self.d2 = Dropout(0.5)  # dropout层
        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.p3(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y


model = Fashion_Model()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])  # 配置好参数，在model.fit时才会运行

checkpoint_save_path = "./teachingcode-lesson8-CNN/checkpoint/fashion_alexnet.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,  # false的话存权重+模型
                                                 # monitor='loss',  # 只有loss下降时才存
                                                 # save_best_only=True,
                                                 verbose=2)  # 显示回调，以前是用for循环每层save权重
history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=16), epochs=5,
                    validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cp_callback], verbose=1)  # verbose=0不在标准流输出训练信息；verbose=1打印进度条；verbose=2打印每个epoch一行

model.summary()

file = open('./teachingcode-lesson8-CNN/fashione_alexnet_weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
