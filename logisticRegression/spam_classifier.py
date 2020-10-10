# -*- coding : utf+2 -*-
# %%

import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer  # 从Sklearn特征提取的文本模块-把Tf词频idf变成向量

df = pd.read_csv('SMSSpamCollection.txt', delimiter='\t', header=None)  # 分隔符为一个缩近，没有文件头
y, X_train = df[0], df[1]

vectorizer = TfidfVectorizer() # 使用前需要进行实例化
X = vectorizer.fit_transform(X_train)  # X是一个数值向量,返回所有词语的TF-IDF的值（越大对所在文章越重要） 将训练数据转化为


lr = linear_model.LogisticRegression()
lr.fit(X, y)  # 进行训练

test = ['URGENT! Your mobile No.1233 was awarded a prize',
                                  'Hey honey, whats up?']

testX = vectorizer.transform(test)  # 用训练数据得到的矢量化器将测试字符串转化为特征向量

predictions = lr.predict(testX)  # 进行预测
print(testX)
print(test[0], '\tis', predictions[0])
print(test[1], '\tis', predictions[1])



