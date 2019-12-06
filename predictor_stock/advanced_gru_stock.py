# coding:utf-8
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN,Input,LSTM,GRU
from tensorflow.keras import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
maotai = pd.read_csv('SH600519.csv')

training_set = maotai.iloc[0:2186 - 300, 1:6].values
test_set = maotai.iloc[2186 - 300:, 1:6].values
# print(training_set.shape)
# print(training_set)
# 归一化
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.transform(test_set)

# 标签归一化,提取sc_label参数解决训练数据(5列)和标签列数(1列)不同的导致反归一化出错问题，用于后续标签反归一化，作股价预测值与真实值的对比图时用，不对实际训练产生影响
# test_set_scaled中的标签列和test_set_label_scaled值是一样的
sc_label = MinMaxScaler(feature_range=(0, 1))
traing_set_label_scaled = sc_label.fit_transform(training_set[:,0:1])
test_set_label_scaled = sc_label.transform(test_set[:,0:1])



x_train = []
y_train = []

x_test = []
y_test = []

#                     2186-300
# 前60天的数据当做输入x,第61天数据当作目标y
for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 60:i, 0:5])
    y_train.append(training_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# print(x_train.shape)
# print(y_train.shape)
# print('x_train:', x_train)
# print('x_train:', x_train.shape)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 5))

for i in range(60, len(test_set)):
    x_test.append(test_set_scaled[i - 60:i, 0:5])
    y_test.append(test_set_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 5))

# print('x_train:', x_train)
print('x_train:', x_train.shape)


class GRUModel(Model):
    def __init__(self):
        super(GRUModel,self).__init__()
        self.l1 = GRU(512,activation='relu',return_sequences=False,unroll=True)
        self.d1 = Dropout(0.2)
        self.f1 = Dense(1)
    def call(self,x):
        x = self.l1(x)
        x = self.d1(x)
        y = self.f1(x)
        return y

model = GRUModel()

model.compile(optimizer='adam',
              loss='mean_squared_error')

checkpoint_save_path = "drive/My Drive/teachingcode-lesson10/stock_gru_byj_4.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 verbose=2)

history = model.fit(x_train, y_train, epochs=200, batch_size=64, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback], verbose=2,shuffle=True)



model.summary()

model.evaluate(x_test,y_test)

# file = open('./weights.txt', 'w')  # 参数提取
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()



loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

################## predict ######################

predicted_stock_price = model.predict(x_test)
# 对预测数据还原。
predicted_stock_price = sc_label.inverse_transform(predicted_stock_price)

# print(predicted_stock_price)
# print(test_set[60:])

real_stock_price = sc_label.inverse_transform(test_set_scaled[60:,0:1])

plt.plot(real_stock_price, color='red', label='MaoTai Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted MaoTai Stock Price')
plt.title('MaoTai Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('MaoTai Stock Price')
plt.legend()
plt.show()