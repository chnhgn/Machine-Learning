# -*- coding: utf-8 -*-
'''
Created on Sep 18, 2018

@author: Eddy Hu
'''


import pandas as pd


train = pd.read_csv('C:\\scnguh\\datamining\\NYC taxi trip duration\\train.csv')
test = pd.read_csv('C:\\scnguh\\datamining\\NYC taxi trip duration\\test.csv')

# print(train.info())

X_train = train.drop(['dropoff_datetime', 'trip_duration', 'id'], axis=1)
y_train = train['trip_duration']
X_test = test.drop(['id'], axis=1)

X_train['month'] = pd.DatetimeIndex(X_train.pickup_datetime).month
X_train['day'] = pd.DatetimeIndex(X_train.pickup_datetime).dayofweek
X_train['hour'] = pd.DatetimeIndex(X_train.pickup_datetime).hour
X_train['store_and_fwd_flag'].replace('Y', 1, inplace=True)
X_train['store_and_fwd_flag'].replace('N', 0, inplace=True)
X_train = X_train.drop(['pickup_datetime'], axis=1)

X_test['month'] = pd.DatetimeIndex(X_test.pickup_datetime).month
X_test['day'] = pd.DatetimeIndex(X_test.pickup_datetime).dayofweek
X_test['hour'] = pd.DatetimeIndex(X_test.pickup_datetime).hour
X_test['store_and_fwd_flag'].replace('Y', 1, inplace=True)
X_test['store_and_fwd_flag'].replace('N', 0, inplace=True)
X_test = X_test.drop(['pickup_datetime'], axis=1)

# 使用RandomForestRegressor进行回归预测
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_y_predict = rfr.predict(X_test)

print(rfr_y_predict)


