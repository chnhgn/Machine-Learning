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



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_id = test.id

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
test.drop(['country_destination', 'id'], axis=1, inplace=True)


def gen_result(test_id, yhat):
    s = pd.Series(yhat, name='label')
    df = pd.DataFrame({'id' : test_id, 'label' : s})
    df.label = df.label.astype('int')
    df['country'] = lbl.inverse_transform(df.label)
    df.drop(['label'], axis=1, inplace=True)
    return df
    
    
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
    'nthread': 8,  # cpu 线程数
}

watchlist = [(dtrain, 'train')]

bst = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist)

# y_test_pred = bst.predict(dtest_sample)
y_pred = bst.predict(dtest)
result = gen_result(test_id, y_pred)
result.to_csv('submission.csv', index=False)

# print('准确率：%.4f' % (np.sum(y_test_pred == y_test) / len(y_test)))


    









