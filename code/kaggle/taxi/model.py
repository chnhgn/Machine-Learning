# -*- coding: utf-8 -*-
'''
Created on Sep 19, 2018

@author: Eddy Hu
'''

import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')  # to be predicted
test_ori = pd.read_csv('C:\\scnguh\\datamining\\NYC taxi trip duration\\test.csv')
yTrain = np.array(train.trip_duration)
train.drop(['trip_duration'], axis=1, inplace=True)

# 对值较大的特征做缩放有利于提高函数收敛速度
x_train, x_test, y_train, y_test = train_test_split(train, np.log(yTrain), test_size=0.15, random_state=2)

# 设置参数
num_trees = 450
params = {"objective": "reg:linear",
          "eta": 0.15,
          "max_depth": 8,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 1
          }

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)
dForecast = xgb.DMatrix(test)
watchlist = [(dtrain, 'train')]

# 训练模型
gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, verbose_eval=True)

yhat = gbm.predict(dtest)
  
RMSE = np.sqrt(mean_squared_error(y_test, yhat))
  
print(RMSE)

yhat = gbm.predict(dForecast)

# 生成预测集
df_forecast = pd.DataFrame(np.exp(yhat), columns=['trip_duration'])
df_forecast = pd.concat([test_ori['id'], df_forecast], axis=1)
df_forecast.to_csv('forecast.csv', index=False)




