# -*- coding: utf-8 -*-
'''
Created on Nov 7, 2018

@author: Eddy Hu
'''
import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier)
from sklearn.svm import SVC



train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
test_id = test.id

lbl = preprocessing.LabelEncoder() 
lbl.fit(list(train.country_destination.values)) 
train.country_destination = lbl.transform(list(train.country_destination.values))

yTrain = np.array(train.country_destination)
train.drop(['country_destination', 'id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)


def gen_result(test_id, yhat):
    s = pd.Series(yhat, name='label')
    df = pd.DataFrame({'id' : test_id, 'label' : s})
    df.label = df.label.astype('int')
    df['country'] = lbl.inverse_transform(df.label)
    df.drop(['label'], axis=1, inplace=True)
    return df 

''' 
    xgboost
'''
    
dtrain = xgb.DMatrix(train, label=yTrain)
dtest = xgb.DMatrix(test)

x_train, x_test, y_train, y_test = train_test_split(train, yTrain, test_size=0.30, random_state=2)

dtrain_sample = xgb.DMatrix(x_train, label=y_train)
dtest_sample = xgb.DMatrix(x_test)

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类的问题
    'num_class': 12,  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 9,  # 构建树的深度，越大越容易过拟合
    'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'min_child_weight': 3,
    'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.007,  # 如同学习率
    'seed': 1000,
    'nthread': 4,  # cpu 线程数
}

watchlist = [(dtrain, 'train')]

bst = xgb.train(params, dtrain, num_boost_round=300, evals=watchlist)

# y_test_pred = bst.predict(dtest_sample)
y_pred = bst.predict(dtest)
result = gen_result(test_id, y_pred)
result.to_csv('./data/submission1.csv', index=False)
print('xgboost finished!')

# print('准确率：%.4f' % (np.sum(y_test_pred == y_test) / len(y_test)))


'''
    logistic regression
'''
lr = LogisticRegression().fit(train, yTrain)
yhat = lr.predict(test)
result = gen_result(test_id, yhat)
result.to_csv('./data/submission2.csv', index=False)
print('logistic regression finished!')
    

'''
    random forest
'''
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
classifier.fit(train, yTrain)
y_pred = classifier.predict(test)
result = gen_result(test_id, y_pred)
result.to_csv('./data/submission3.csv', index=False)
print('random forest finished!')

'''
    Ada boost
'''
ada_params = {
    'n_estimators': 200,
    'learning_rate' : 0.75
}

clf = AdaBoostClassifier(**ada_params)
clf.fit(train, yTrain)
y_pred = clf.predict(test)

result = gen_result(test_id, y_pred)
result.to_csv('./data/submission4.csv', index=False)
print('adaboost finished!')

# Vote for the result

res1 = pd.read_csv('./data/submission1.csv')
res2 = pd.read_csv('./data/submission2.csv')
res3 = pd.read_csv('./data/submission3.csv')
res4 = pd.read_csv('./data/submission4.csv')

label1 = np.array(lbl.transform(list(res1.country.values))).reshape(-1, 1)
label2 = np.array(lbl.transform(list(res2.country.values))).reshape(-1, 1)
label3 = np.array(lbl.transform(list(res3.country.values))).reshape(-1, 1)
label4 = np.array(lbl.transform(list(res4.country.values))).reshape(-1, 1)

label_all = np.concatenate((label1, label2, label3, label4), axis=1)

vote = []
for line in label_all:
    vote.append(np.argmax(np.bincount(line)))

result = gen_result(test_id, vote)
result.to_csv('./data/submission_vote.csv', index=False)
print('vote finished!')









