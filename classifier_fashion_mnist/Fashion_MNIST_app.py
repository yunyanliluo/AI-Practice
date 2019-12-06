# -*- coding:utf-8 -*-

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=float('inf'))
model_checkpoint_path = './checkpoint/cp-0020.ckpt'

model = tf.keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')])

model.load_weights(model_checkpoint_path)

app_file_path = './fashion_pic/'
files = os.listdir(app_file_path)
print("app files:", files)

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for file in files:
    print("current picture:", file)
    image_path = app_file_path + file
    image = plt.imread(image_path)
    plt.set_cmap('gray')
    plt.imshow(image)
    plt.pause(2)
    plt.close()

    img = Image.open(image_path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    plt.imshow(img)
    plt.pause(1)
    plt.close()

    img_arr = np.array(img.convert('L'))

    plt.imshow(img_arr)
    plt.pause(1)
    plt.close()

    for i in range(28):
        for j in range(28):
            img_arr[i][j] = 255 - img_arr[i][j]
    plt.imshow(img_arr)
    plt.pause(1)
    plt.close()

    img_arr = img_arr / 255.0

    x_predict = img_arr[tf.newaxis, ...]

    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)
    print("predict label:", labels[int(pred)], '\n')

# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot
