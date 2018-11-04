# -*- coding: utf-8 -*-
'''
Created on Nov 2, 2018

@author: Eddy Hu
'''

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 300)
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing



train = pd.read_csv('all_features_train.csv')
test = pd.read_csv('all_features_test.csv')

dest_key = np.array(train.country_destination)
lbl = preprocessing.LabelEncoder() 
lbl.fit(list(train.country_destination.values)) 
train.country_destination = lbl.transform(list(train.country_destination.values))
dest_value = np.array(train.country_destination)
dest_dict = np.vstack((dest_key, dest_value)).T
dest_dict = pd.DataFrame(dest_dict, columns=['key', 'value'])
dest_dict.drop_duplicates(inplace=True)
dest_dict.reset_index(inplace=True, drop=True)

yTrain = np.array(train.country_destination)
train.drop(['country_destination', 'id'], axis=1, inplace=True)


# # 为字符串值编号int数值
# for col in train.columns: 
#     if train[col].dtype == 'object': 
#         lbl = preprocessing.LabelEncoder() 
#         lbl.fit(list(train[col].values)) 
#         train[col] = lbl.transform(list(train[col].values))
#         
# for col in test.columns: 
#     if test[col].dtype == 'object': 
#         lbl = preprocessing.LabelEncoder() 
#         lbl.fit(list(test[col].values)) 
#         test[col] = lbl.transform(list(test[col].values))

x_train, x_test, y_train, y_test = train_test_split(train, yTrain, test_size=0.30, random_state=2)

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

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
    'nthread': 8,  # cpu 线程数
}

watchlist = [(dtrain, 'train')]

bst = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist)

y_test_pred = bst.predict(dtest)


print('准确率：%.4f' % (np.sum(y_test_pred == y_test) / len(y_test)))


