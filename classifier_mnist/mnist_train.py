from PIL import Image
import numpy as np

import tensorflow as tf

np.set_printoptions(threshold=float('inf'))
model_save_path = './studentcode-course6/checkpoint/mnist.tf'
load_pretrain_model = False

train_path = './studentcode-course6/mnist_image_label/mnist_train_jpg_60000/'
train_txt = './studentcode-course6/mnist_image_label/mnist_train_jpg_60000.txt'

test_path = './studentcode-course6/mnist_image_label/mnist_test_jpg_10000/'
test_txt = './studentcode-course6/mnist_image_label/mnist_test_jpg_10000.txt'


# 自制数据集（处理输入图片和输入标签）
def generateds(path, txt):  # 输入图片目录，输入标签文件名
    f = open(txt, 'r')
    contents = f.readlines()  # 按行读取
    f.close()
    images, labels = [], []
    for content in contents:
        value = content.split()  # 以空格分开，存入数组
        img_path = path + value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L'))
        img = img / 255.
        images.append(img)
        labels.append(value[1])
        print('loading : ' + content)

    x_ = np.array(images)
    y_ = np.array(labels)
    y_ = y_.astype(np.int64)
    return x_, y_


print('-------------load the data-----------------')
x_train, y_train = generateds(train_path, train_txt)
x_test, y_test = generateds(test_path, test_txt)


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

if load_pretrain_model:
    print('-------------load the model-----------------')
    model.load_weights(model_save_path)

# save_model_1
# 训完后保存模型+可读取模型续训
model.fit(x_train, y_train, epochs=2, validation_data=(x_test,y_test),validation_freq=2)
model.summary()

model.save_weights(model_save_path, save_format='tf')

# save_model_2
# 每训1个epoch保存一次模型+可读取模型续训
# model.summary()
# for i in range(5):
#     model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), validation_freq=2)
#     model.save_weights(model_save_path, save_format='tf')

file = open('studentcode-course6/weights', 'w')
for v in model.trainable_variables:
    file.write(v.name + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()
