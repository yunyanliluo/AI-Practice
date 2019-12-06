# 增加自制数据集，利用class结构训练并测试cifar10

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout, Flatten,Dense
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import os
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.set_printoptions(threshold=float('inf'))
model_save_path = './teachingcode-lesson7_code/checkpoint/cifar10.tf'


train_path = './teachingcode-lesson7_code/cifar-10/cifar-10/train/'
test_path = './teachingcode-lesson7_code/cifar-10/cifar-10/test/'


# 自制数据集（处理输入图片和输入标签）
def generateds(path):  # 输入图片目录, 目录序号就是标签
    contents = os.listdir(path)
    images, labels = [], []
    for content in contents:
        img_dir = path + content
        pictures = os.listdir(img_dir)
        for picture in pictures:
            img_path = img_dir + '/' + picture
            img = Image.open(img_path)
            img = np.array(img.convert('RGB'))
            img = img / 255
            images.append(img)
            labels.append(contents.index(content))
            print('loading : ' + img_path)

    x_ = np.array(images)
    y_ = np.array(labels)
    y_ = y_.astype(np.int64)

    # 打乱数据集
    np.random.seed(116)
    np.random.shuffle(x_)
    np.random.seed(116)
    np.random.shuffle(y_)

    return x_, y_


print('-------------load the data-----------------')
x_train, y_train = generateds(train_path)
x_test, y_test = generateds(test_path)
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train=tf.convert_to_tensor(x_train)
x_test=tf.convert_to_tensor(x_test)
# y_train=tf.squeeze(y_train, axis=1)
# y_test=tf.squeeze(y_test, axis=1)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


image_gen_train = ImageDataGenerator(
                                     # rescale=1./255,#归至0～1
                                     rotation_range=45,#随机45度旋转
                                     width_shift_range=.15,#宽度偏移
                                     height_shift_range=.15,#高度偏移
                                     horizontal_flip=True,#水平翻转
                                     zoom_range=0.5#将图像随机缩放到50％
                                     )
image_gen_train.fit(x_train)


class Cifar10_Model(Model):
    def __init__(self):
        super(Cifar10_Model, self).__init__()
        self.c1 = Conv2D(input_shape=(32, 32, 3), filters=32, kernel_size=(5, 5), padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.m1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.drop1 = Dropout(0.2)  # dropout层

        self.c2 = Conv2D(64, kernel_size=(5, 5), padding='same')
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.m2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.drop2 = Dropout(0.2)

        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.drop3 = Dropout(0.2)
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.m1(x)
        x = self.drop1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.m2(x)
        x = self.drop2(x)

        x = self.flatten(x)
        x = self.d1(x)
        x = self.drop3(x)
        y = self.d2(x)
        return y


model = Cifar10_Model()


# model = tf.keras.models.Sequential([
#     Conv2D(filters=32,kernel_size=(5,5),padding='same',input_shape=(32,32,3)),
#     BatchNormalization(),
#     Activation('relu'),
#     MaxPool2D(pool_size=(2,2),strides=2,padding='same'),
#     Dropout(0.2),
#
#     Conv2D(64, kernel_size=(5,5), padding='same'),
#     BatchNormalization(),
#     Activation('relu'),
#     MaxPool2D(pool_size=(2,2), strides=2, padding='same'),
#     Dropout(0.2),
#
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.2),
#     Dense(10, activation='softmax')
# ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

if os.path.exists(model_save_path+'.index'):
    print('-------------load the model-----------------')
    model.load_weights(model_save_path)

# save_model_1
# model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_test, y_test), validation_freq=2)
# model.summary()
# model.save_weights(model_save_path, save_format='tf')

# save_model_2
for i in range(1):
    history = model.fit(image_gen_train.flow(x_train,y_train), epochs=1, batch_size=32, validation_data=(x_test, y_test), validation_freq=1)
    model.save_weights(model_save_path, save_format='tf')
model.summary()

file = open('./teachingcode-lesson7_code/cifar10_weights.txt', 'w')  # 参数提取
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



# 图片增强的对比显示
x_train_subset1 = np.squeeze(x_train[:12])
x_train_subset2 = x_train[:12]  # 一次显示12张图片

fig = plt.figure(figsize=(20,2))
plt.set_cmap('gray')
# 显示原始图片
for i in range(0, len(x_train_subset1)):
    ax = fig.add_subplot(1, 12, i+1)
    ax.imshow(x_train_subset1[i])
fig.suptitle('Subset of Original Training Images', fontsize=20)
plt.show()

# 显示增强后的图片
fig = plt.figure(figsize=(20,2))
for x_batch in image_gen_train.flow(x_train_subset2, batch_size=12,shuffle=False):
    for i in range(0, 12):
        ax = fig.add_subplot(1, 12, i+1)
        ax.imshow(np.squeeze(x_batch[i]))
    fig.suptitle('Augmented Images', fontsize=20)
    plt.show()
    break;
