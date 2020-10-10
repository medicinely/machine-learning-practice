# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 21:29:25 2019

@author: Leo
"""

# 鸢尾花分类
import numpy as np
import pandas as pd
from keras.models import Sequential  # 序列
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score  # 交叉验证
from sklearn.model_selection import KFold  # 将数据集分成K个小数据集（每次选一个作为测试集）重复k词
from sklearn.preprocessing import LabelEncoder  # 编码， 将数据集中文本转为数字
from keras.models import model_from_json  # 保存模型

# reproducibility 可重复性
seed = 13
np.random.seed(seed)  # 随机生成一个矩阵，random.seed代表以后每次生成的数字都为同一个，不再变化

# load data
# 花萼长度，花萼宽度，花瓣长度，花瓣宽度，类别
df = pd.read_csv('iris.csv')  # df(data frame)
X = df.values[:, 0:4].astype(float)  # values返回值，float占用32字节（精度已经足够# ），double占用64字节
Y = df.values[:, 4]  # 类别（文字）

encoder = LabelEncoder()  # 类别（文字转编码）
Y_encoded = encoder.fit_transform(Y)  # fit训练 transform转换
Y_onehot = np_utils.to_categorical(Y_encoded)


# onehot-独热编码，将分类变量作为二进制表示
# np_utils.to_categorical用于将整形变量转化为二进制变量

# define a '基础模型' network 方法（函数）
def baseline_model():
    model = Sequential()  # 这个模型的对象（特征）是顺序的
    model.add(Dense(7, input_dim=4, activation='tanh'))
    # 在model里加入 输入层-隐含层，有7个node（节点），输入dimension是4，激活函数是tanh（hyperbolic tan）
    model.add(Dense(3, activation='softmax'))  # model中加入输出层，dimension是3，激活函数是softmax
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    # 编译这个模型，指定loss-function=均方差（衡量网络输出和真实结果的差值），
    # 优化器=Stochastic gradient descent随机梯度下降（用什么方法来训练网络进步）
    # metrics（怎样衡量模型好坏）= 准确度衡量（用百分比衡量 ）
    return model


# Keras分类（模型名称，训练次数，每次训练的数据个数，verbose：0.不显示进度条/1.显示进度条/2.每个epoch显示一个进度条/）
# 利用这个类我们就可以方便的调用sklearn包中的一些函数进行数据预处理和结果评估
estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=1, verbose=2)


# evaluate
# KFold类，一种交叉验证方法（n_splits把数据分成10份，shuffle打乱，重复）
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)  # 十次交叉验证（9次用来验证）
result = cross_val_score(estimator, X, Y_onehot, cv=kfold)
print('Accuracy of cross validation, mean %.2f, std %.2f' % (result.mean(), result.std())) # 打印均值方差

# save model
estimator.fit(X, Y_onehot)
model_json = estimator.model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)   # 保存模型

estimator.model.save_weights('model.h5')  # 存入权重文件
print('saved model to disk')

# load model and use it for prediction
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()  # 读入模型
json_file.close()

loaded_model = model_from_json(loaded_model_json)  # model_from_json加载模型
loaded_model.load_weights('model.h5')  # 读取权重文件
print('loaded model from disk')


predicted = loaded_model.predict(X)
print('predicted probability:' + str(predicted))

predicted_label = loaded_model.predict_classes(X)
print('predicted label:' + str(predicted_label))
