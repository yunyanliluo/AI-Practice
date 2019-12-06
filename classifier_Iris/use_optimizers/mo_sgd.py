# 有momentum的SGD

import tensorflow as tf
import os
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error，不显示其他信息


def normalize(train):
    # 线性归一化
    x_data = train.T
    for i in range(4):
        x_data[i] = (x_data[i] - tf.reduce_min(x_data[i])) / (tf.reduce_max(x_data[i]) - tf.reduce_min(x_data[i]))
    return x_data.T


def norm_nonlinear(train):
    # 非线性归一化（log）
    x_data = train.T
    for i in range(4):
        x_data[i] = np.log10(x_data[i]) / np.log10(tf.reduce_max(x_data[i]))
    return x_data.T


def standardize(train):
    # 数据标准化（标准正态分布）
    x_data = train.T
    for i in range(4):
        x_data[i] = (x_data[i] - np.mean(x_data[i])) / np.std(x_data[i])
    return x_data.T


x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

x_data = standardize(x_data)

# 随机打乱数据
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)

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

# from_tensor_slices函数切分传入的 Tensor 的第一个维度，生成相应的 dataset
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(10)

# iter用来生成迭代器
train_iter = iter(train_db)
# next() 返回迭代器的下一个项目
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)

# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

learning_rate_step = 10
learning_rate_decay = 0.8
train_loss_results = []
test_acc = []
lr = []
global_step = epoch = 500
loss_all = 0
learning_rate_base = 1
m_w, m_b = 0, 0

now_time = time.time()
for epoch in range(epoch):
    learning_rate = learning_rate_base * learning_rate_decay ** (epoch / learning_rate_step)
    lr.append(learning_rate)
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            # y = tf.math.tanh(y)
            y_onehot = tf.one_hot(y_train, depth=3)
            # # mse = mean(sum(y-out)^2)
            # loss = tf.reduce_mean(tf.square(y_onehot - y))
            # cross-entropy
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_onehot, y))
            loss_all += loss.numpy()
        # compute gradients
        grads = tape.gradient(loss, [w1, b1])

        # sgd-momentun
        beta = 0.9
        m_w = beta * m_w + (1 - beta) * grads[0]
        m_b = beta * m_b + (1 - beta) * grads[1]
        w1.assign_sub(learning_rate * m_w)
        b1.assign_sub(learning_rate * m_b)

        if step % 10 == 0:
            print("step=", step, 'loss:', float(loss))
            print("lr=", learning_rate)
    train_loss_results.append(loss_all / 3)
    loss_all = 0

    # test(做测试）
    total_correct, total_number = 0, 0
    for step, (x_test, y_test) in enumerate(test_db):
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.sigmoid(y)

        pred = tf.argmax(y, axis=1)

        # 因为pred的dtype为int64，在计算correct时会出错，所以需要将它转化为int32
        pred = tf.cast(pred, dtype=tf.int64)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int64)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]
    acc = total_correct / total_number
    test_acc.append(acc)
    print("test_acc:", acc)
    print("---------------------")
total_time = time.time() - now_time
print("total_time", total_time)

# 绘制 loss 曲线
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label="$Loss$")
plt.legend()
plt.show()

# 绘制 Accuracy 曲线
plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()

# 绘制 Learning_rate 曲线
plt.title('Learning Rate Curve')
plt.xlabel('Global steps')
plt.ylabel('Learning rate')
plt.plot(range(global_step), lr, label="$lr$")
plt.legend()
plt.show()
