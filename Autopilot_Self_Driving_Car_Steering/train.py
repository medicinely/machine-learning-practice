# -*- coding: utf-8 -*-
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
#控制显存占用：动态增长
'''
import time
import csv
import cv2
import glob
import json
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import callbacks
from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, PReLU
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

start = time.process_time()

def horizontal_flip(img, degree):
    # 按照50%的概率水平翻转图像
    choice = np.random.choice([0, 1])
    if choice == 1:
        img, degree = cv2.flip(img, 1), -degree
    return (img, degree)


def random_brightness(img, degree):

    # 随机调整输入图像的亮度（以模拟白天和夜晚），调整强度于 0.1(变黑)和1(无变化)之间
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # HSV 色相 饱和度 亮度
    # 调整亮度 V
    alpha = np.random.uniform(low=0.1, high=1.0, size=None)
    v = hsv[:, :, 2]
    v = v * alpha
    hsv[:, :, 2] = v.astype('uint8')
    rgb = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)
    return (rgb, degree)



def left_right_random_swap(img_address, degree, degree_corr=0.25):
    # 随机从左， 中， 右图像中选择一张图像， 并相应调整转动的角度
    swap = np.random.choice(['L', 'R', 'C'])
    if swap == 'L':
        img_address = img_address.replace('center', 'left')
        corrected_degree = np.arctan(math.tan(degree) + degree_corr)
        return (img_address, corrected_degree)
    elif swap == 'R':
        img_address = img_address.replace('center', 'right')
        corrected_degree = np.arctan(math.tan(degree) - degree_corr)
        return (img_address, corrected_degree)
    else:
        return (img_address, degree)


def image_transformation(img_address, degree, data_dir):

    # 调用上述函数完成图像预处理：选图、调亮度、水平翻转
    img_address, degree = left_right_random_swap(img_address, degree)
    img = cv2.imread(data_dir + img_address)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, degree = random_brightness(img, degree)
    img, degree = horizontal_flip(img, degree)
    return (img, degree)


def discard_zero_steering(degrees, rate):
    # 从角度为零的index中随机选择部分index返回，丢弃
    steering_zero_idx = np.where(degrees == 0)[0]
    size_del = int(len(steering_zero_idx) * rate)
    # 选出size_del个元素丢弃:
    return np.random.choice(steering_zero_idx, size=size_del, replace=False)


def get_model(shape):  # 创建网络
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation='relu', input_shape=shape))
    # 5x5卷积核 1x1步长 填充方式：same 激活韩函数：relu 第一层输入的图像储存=图像本身shape（第二层不用指定）
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # 将前面所有网络结构变成一维向量才能和后面层连接

    model.add(Dense(256, activation='relu'))  # Dense是Keras里面的全连接层
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))  # 因为输出含有负数，激活函数要使用linear

    model.compile(optimizer=Adam(lr=1e-3), loss='mean_squared_error')
    # 编译，通过'optimizer：Adam（learing rates学习率）'优化器减小损失函数，如果学习率太大，损失函数可能不会收敛到理想值
    # 损失函数
    return model


def batch_generator(x, y, batch_size, shape, training=True, data_dir='',
                    monitor=True, yieldXY=True, discard_rate=0.8):
    """
    x,y:文件名，角度
    batch_size:一批数据的图片数量
    shape：网络所需输入图像的尺寸
    training: True产生训练数据，False产生validation（原数据）数据
    monitor: 保存一个batch样本： 'X_batch_sample.npy'，'y_bag.npy’
    yieldXY: True返回(X, Y)，False返回 X 
    discard_rate: 随机丢弃角度为零的训练数据的概率
    """
    if training:
        y_bag = []
        x, y = shuffle(x, y) # 打乱数据
        rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
        new_x = np.delete(x, rand_zero_idx, axis=0)
        new_y = np.delete(y, rand_zero_idx, axis=0)
    else:  # train要删数据，vail不删
        new_x = x
        new_y = y
    offset = 0  # 每完成一次循环增加 batch_size
    while True:
        X = np.empty((batch_size, *shape))  # *号连接 tuple和 int shape(66,200,3) *shape -> Np.empty((16,66,200,3))
        Y = np.empty((batch_size, 1))  # X存放图片，Y存放标签
        for example in range(batch_size):
            img_address, img_steering = new_x[example + offset], new_y[example + offset]
            # 提醒自己检查图片路径是否正确
            #print('address is:',data_dir + img_address)
            if training:
                img, img_steering = image_transformation(img_address, img_steering, data_dir)
            else:
                img = cv2.imread(data_dir + img_address)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X[example, :, :, :] = cv2.resize(img[80:140, 0:320], (shape[0], shape[1])) / 255 - 0.5
            # X[第几个,高度,宽度,RGB通道几]
            Y[example] = img_steering
            if monitor:
                y_bag.append(img_steering)

            # 到达原来数据的结尾后, 从头开始
            if (example + 1) + offset > len(new_y) - 1:
                x, y = shuffle(x, y)
                rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
                new_x = x
                new_y = y
                new_x = np.delete(new_x, rand_zero_idx, axis=0)
                new_y = np.delete(new_y, rand_zero_idx, axis=0)
                offset = 0
        if yieldXY:
            yield (X, Y)  # 返回值
        else:
            yield X
        offset = offset + batch_size
        if monitor:
            np.save('y_bag.npy', np.array(y_bag))
            np.save('Xbatch_sample.npy', X)


if __name__ == '__main__':
    SEED = 13
    data_path = 'data/'
    with open(data_path + 'driving_log.csv', 'r') as csvfile:

        file_reader = csv.reader(csvfile, delimiter=',')
        log = []
        for row in file_reader:
            log.append(row)

    log = np.array(log)
    # 去掉文件第一行
    #log = log[1:, :]
    print(log.shape)

    # 判断图像文件数量是否等于csv日志文件中记录的数量（glob：Regular expression正则表达式
    # ）
    ls_imgs = glob.glob(data_path + 'IMG/*.jpg')
    assert len(ls_imgs) == len(log) * 3, 'number of images does not match'

    # 使用20%的数据作为validation测试数据
    validation_ratio = 0.2
    shape = (128, 128, 3)
    # 一批数量batch size
    #  batch_size = 128
    batch_size = 8
    nb_epoch = 30  # 选择训练n个epoch

    x_ = log[:, 0] #把csv数据中的x和y赋值给x_
    y_ = log[:, 3].astype(float)
    x_, y_ = shuffle(x_, y_) #洗牌
    X_train, X_val, y_train, y_val = train_test_split(x_, y_, test_size=validation_ratio, random_state=SEED)
    #根据我的validation ratio 把训练数据和测试数据分开


    print('batch size: {}'.format(batch_size))
    print('Train set size: {} | Validation set size: {}'.format(len(X_train), len(X_val)))

    samples_per_epoch = batch_size # 每一个epoch中训练多少次
    # 使得validation数据量大小为batch_size的整数陪
    nb_val_samples = len(y_val) - len(y_val) % batch_size
    # 定义网路结构
    model = get_model(shape)
    # 打印网络层
    print(model.summary())

    # keras/tensorflow有常用的'callback'把传进去的函数在中间过程做一些事情，根据validation loss保存最优模型
    # ModelCheckpoint：把训练过程中最好的模型存下来
    save_best = callbacks.ModelCheckpoint('model.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='auto')

    # 如果训练持续没有validation loss的提升, 提前结束训练
    # Earlystopping：如果没有变好就提前停止 patience(向后看多少步) min_delta(和前一次进步小于多少的时候停止)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15,
                                         verbose=0, mode='auto')
    # TensorBoard 可视化训练过程，不再是一个黑盒子
    # tbCallBack = callbacks.TensorBoard(log_dir='./Graph',write_graph=True,write_images=True)
    # callbacks_list = [early_stop, save_best, tbCallBack]
    callbacks_list = [early_stop, save_best]
    print(nb_val_samples // batch_size)
    # 真正开始训练！fit_generator(数据，每个训练过程多少步，validation的step数量，传入validation的数据，verbose把中间过程打印出来)
    history = model.fit_generator(batch_generator(X_train, y_train, batch_size, shape, training=True, monitor=False),
                                  steps_per_epoch=samples_per_epoch,
                                  validation_steps=nb_val_samples // batch_size,# 测试总数据整除每一次的数量
                                  validation_data=batch_generator(X_val, y_val, batch_size, shape,
                                                                  training=False, monitor=False),
                                  epochs=nb_epoch,
                                  verbose=1,
                                  callbacks=callbacks_list)
    # 创建数据字典'***.p'保存训练结果history（要保存history.histoty才不会报错）
    with open('./trainHistoryDict.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('train_val_loss.jpg', dpi=600)

    # 保存模型
    with open('model.json', 'w') as f:
        f.write(model.to_json()) # 把模型架构保存成json文件
    model.save('model.h5')   # 也可以保存模型到'****.h5'文件
    print('Done!')

end = time.process_time()
print('执行时间:%6.3f', end - start)