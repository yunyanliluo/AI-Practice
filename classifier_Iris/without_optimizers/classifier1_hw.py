# -*- coding: UTF-8 -*-
import math

import tensorflow as tf
import os
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error，不显示其他信息

# 从文件导入数据
iris_data = pd.read_csv('第三次课作业/参考文件/iris.txt', header=None)  # 读取数据,不把第一行作为列属性
iris_data = iris_data.replace('Iris-setosa', 0)
iris_data = iris_data.replace('Iris-versicolor', 1)
iris_data = iris_data.replace('Iris-virginica', 2)
print(iris_data.head())  # 查看前5行数据

# 预处理数据
x_data = iris_data.iloc[:, 0:4]  # 取 pandas DataFrame 的0-3列
y_data = iris_data.iloc[:, 4:]  # 取 pandas DataFrame 的4列
x_data = x_data.values  # pandas DataFrame 转 np.array
y_data = np.squeeze(y_data.values)  # 从np.array的形状中删除单维条目，即把shape中为1的维度去掉
y_data = y_data.astype('int32')  # np.array change dtype
print(x_data)
print(y_data)

# 随机打乱数据,不然按照先训练标签0的，再训练标签1的，最后训练2的，训练结果最后永远是2
np.random.seed(116)
np.random.shuffle(x_data)  # 先将x以随机数116打乱一次
np.random.seed(116)
np.random.shuffle(y_data)  # 再将y以随机数116打乱一次

x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

print("x.shape:", x_data.shape)
print("y.shape:", y_data.shape)
print("x.dtype:", x_data.dtype)
print("y.dtype:", y_data.dtype)
print("min of x:", tf.reduce_min(x_data))
print("max of x:", tf.reduce_max(x_data))
print("min of y:", tf.reduce_min(y_data))
print("max of y:", tf.reduce_max(y_data))

# from_tensor_slices函数切分传入的 Tensor 的第一个维度（把data与data之间切分开），生成相应的 dataset
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(10)  # 一次喂10个数据
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(10)  # 一次喂10个数据（测试的时候随机选10个）
# iter用来生成迭代器
train_iter = iter(train_db)
# next() 返回迭代器的下一个项目
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)

# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样 #可训练参数就是用tf.Variable引导的
w1 = tf.Variable(tf.random.truncated_normal([4, 32], stddev=0.1, seed=1))  # 输入的x是四维的
b1 = tf.Variable(tf.random.truncated_normal([32], stddev=0.1, seed=1))
# w2 = tf.Variable(tf.random.truncated_normal([32, 32], stddev=0.1, seed=2))
# b2 = tf.Variable(tf.random.truncated_normal([32], stddev=0.1, seed=2))
w3 = tf.Variable(tf.random.truncated_normal([32, 3], stddev=0.1, seed=3))
b3 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=3))

lr = 0.1
train_loss_results = []
test_accuracy_results = []
epoch = 200
loss_all = 0
for epoch in range(epoch):  # 循环数据集，一个epoch表示对数据集训练一遍
    for step, (x_train, y_train) in enumerate(train_db):  # 循环喂数据，每次一个batch

        with tf.GradientTape() as tape:
            h1 = tf.matmul(x_train, w1) + b1
            # h1 = tf.nn.softmax(h1)
            # h2 = tf.matmul(h1, w2) + b2
            y = tf.matmul(h1, w3) + b3  # 120行，3列
            y = tf.nn.softmax(y)  # 加一个softmax激活函数

            y_onehot = tf.one_hot(y_train, depth=3)  # 120行，3列

            # mse = mean(sum(y-out)^2)
            loss = tf.reduce_mean(tf.square(y_onehot - y))
            loss_all += loss.numpy()

        # compute gradients
        grads = tape.gradient(loss, [w1, b1, w3, b3])  # 对所有参数求偏导数
        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        # w2.assign_sub(lr * grads[2])
        # b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[2])
        b3.assign_sub(lr * grads[3])

        if step % 100 == 0:
            print('epoch:', epoch, 'step:', step, 'loss:', float(loss))
    train_loss_results.append(loss_all / 3)
    loss_all = 0

    # test(做测试）
    total_correct, total_number = 0, 0
    for step, (x_test, y_test) in enumerate(test_db):
        h1 = tf.matmul(x_test, w1) + b1
        # h1 = tf.nn.softmax(h1)
        # h2 = tf.matmul(h1, w2) + b2
        y = tf.matmul(h1, w3) + b3
        y = tf.nn.softmax(y)

        pred = tf.argmax(y, axis=1)  # 根据axis取值的不同返回每行或者每列最大值的索引

        # 因为pred的dtype为int64，在计算correct时会出错，所以需要将它转化为int32
        pred = tf.cast(pred, dtype=tf.int32)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)  # tf.equal逐个元素进行判断，如果相等就是True，不相等，就是False。
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]
        # print("correct:", int(correct), "x_test.shape[0]:", x_test.shape[0])
    acc = total_correct / total_number
    test_accuracy_results.append(acc)
    print("test_acc:", acc)

# 绘制loss曲线
plt.figure(1)
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results)

# 绘制test_acc曲线
plt.figure(2)
plt.title('Test Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Test_acc')
plt.plot(test_accuracy_results)
plt.show()
