import tensorflow as tf
import os
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import time
import pandas as pd


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error，不显示其他信息

def normalize(data):
    # 线性归一化
    x_data = data.T
    for i in range(4):
        x_data[i] = (x_data[i] - tf.reduce_min(x_data[i])) / (tf.reduce_max(x_data[i]) - tf.reduce_min(x_data[i]))
    return x_data.T


def norm_nonlinear(data):
    # 非线性归一化（log）
    x_data = data.T
    for i in range(4):
        x_data[i] = np.log10(x_data[i]) / np.log10(tf.reduce_max(x_data[i]))
    return x_data.T


def standardize(data):
    # 数据标准化（标准正态分布）
    x_data = data.T
    for i in range(4):
        x_data[i] = (x_data[i] - np.mean(x_data[i])) / np.std(x_data[i])
    return x_data.T

#从txt文件读取数据
df = pd.read_csv('iris.txt',header = None,sep=',')
data = df.values
print(data)
x_data = [lines[0:4] for lines in data]
x_data = np.array(x_data,float)
print(x_data)
y_data = [lines[4] for lines in data]
for i in range(len(y_data)):
    if y_data[i] == 'Iris-setosa':
        y_data[i] = 0
    elif y_data[i] == 'Iris-versicolor':
        y_data[i] = 1
    elif y_data[i] == 'Iris-virginica':
        y_data[i] = 2
y_data = np.array(y_data)
print(y_data)
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

# print("x.shape:", x_data.shape)
# print("y.shape:", y_data.shape)
# print("x.dtype:", x_data.dtype)
# print("y.dtype:", y_data.dtype)
# print("min of x:", tf.reduce_min(x_data))
# print("max of x:", tf.reduce_max(x_data))
# print("min of y:", tf.reduce_min(y_data))
# print("max of y:", tf.reduce_max(y_data))

# from_tensor_slices函数切分传入的 Tensor 的第一个维度，生成相应的 dataset
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)

# iter用来生成迭代器
train_iter = iter(train_db)
# next() 返回迭代器的下一个项目
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)

# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))



# lr = 0.1

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.3
decay_steps = 200
decay_rate = 0.8
train_loss_results = []
test_acc = []
lr_ = []
epoch = 1000
loss_all = 0

now_time = time.time()
for epoch in range(epoch):
    # print("epoch", epoch)

    for step, (x_train, y_train) in enumerate(train_db):
        global_step = global_step.assign_add(1)
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y_onehot = tf.one_hot(y_train, depth=3)
            # mse = mean(sum(y-out)^2)
            loss = tf.reduce_mean(tf.square(y_onehot - y))
            loss_all += loss.numpy()
        # compute gradients
        grads = tape.gradient(loss, [w1, b1])
        # w1 = w1 - lr * w1_grad
        learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, decay_steps,
                                                             decay_rate, staircase=False)
        lr = learning_rate()
        # lr = 0.1
        lr_.append(lr)
        # print('lr=', lr)
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        # if step % 2 == 0:
        #     print("step=", step, 'loss:', float(loss))
    train_loss_results.append(loss_all / 3)
    loss_all = 0

    # test(做测试）
    total_correct, total_number = 0, 0
    for step, (x_test, y_test) in enumerate(test_db):

        y = tf.matmul(x_test, w1) + b1
        # hy = tf.nn.leaky_relu(y)
        pred = tf.argmax(y, axis=1)
        # 因为pred的dtype为int64，在计算correct时会出错，所以需要将它转化为int32
        pred = tf.cast(pred, dtype=tf.int32)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
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
plt.subplot(2,2,1)
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label="$Loss$")
plt.legend()


# 绘制 Accuracy 曲线
plt.subplot(2,2,2)
plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label="$Accuracy$")
plt.legend()


# 绘制 Learning_rate 曲线
plt.subplot(2,1,2)
plt.title('Learning Rate Curve')
plt.xlabel('Global steps')
plt.ylabel('Learning rate')
plt.plot(range(global_step.numpy()),lr_, label="$lr$")
plt.legend()
plt.show()
