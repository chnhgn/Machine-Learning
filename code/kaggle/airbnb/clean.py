# -*- coding: utf-8 -*-
import pandas as pd
import os
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt

raw_path = 'C:\\scnguh\\datamining\\airbnb\\all\\'
raw_train = raw_path + 'train_users_2.csv'
raw_test = raw_path + 'test_users.csv'
raw_session = raw_path + 'sessions.csv'

if os.path.exists('data') is False:
    os.mkdir('data')
    
df_raw_train = pd.read_csv(raw_train)
df_raw_test = pd.read_csv(raw_test)
df_raw_session = pd.read_csv(raw_session)

# 合并处理train和test数据集
df_raw = pd.concat([df_raw_train, df_raw_test])

# 性别缺失严重
df_raw.gender = df_raw.gender.apply(lambda x:'unknown' if x == '-unknown-' or x == 'OTHER' else x)
# dummies_features = pd.get_dummies(df_raw.gender, prefix='gender')
# for dummy in dummies_features:
#     df_raw[dummy] = dummies_features[dummy]
#     
# df_raw.drop('gender', inplace=True, axis=1)

# 处理age缺失值
df_raw.age = df_raw.age.apply(lambda x:np.NaN if x > 140 else x)
# print(df_raw.age.isnull().sum())
# print(df_raw.age.describe())
# print(df_raw.age.mean())
df_raw.age.fillna(df_raw.age.mean(), inplace=True)

# 处理first_browser
df_raw.first_browser = df_raw.first_browser.apply(lambda x:'unknown' if x == '-unknown-' else x)
# print(df_raw.groupby(['first_browser'])['first_browser'].count())

# 处理session数据集
# print(df_raw_session.groupby(['action_type'])['action_type'].count())
df_raw_session.secs_elapsed.fillna(0, inplace=True)
df_raw_session.action = df_raw_session.action.apply(lambda x:'unknown' if x == '-unknown-' else x)
df_raw_session.action.fillna('unknown', inplace=True)
df_raw_session.action_type = df_raw_session.action_type.apply(lambda x:'unknown' if x == '-unknown-' else x)
df_raw_session.action_type.fillna('other', inplace=True)
df_raw_session.action_detail = df_raw_session.action_detail.apply(lambda x:'unknown' if x == '-unknown-' else x)
df_raw_session.action_detail.fillna('other', inplace=True)
df_raw_session.device_type = df_raw_session.device_type.apply(lambda x:'unknown' if x == '-unknown-' else x)

# 拆分数据集并保存
train_clean = df_raw[:-len(df_raw_test)]
test_clean = df_raw[-len(df_raw_test):]

train_clean.to_csv(raw_path + 'train_users_clean.csv', index=False)
test_clean.to_csv(raw_path + 'test_users_clean.csv', index=False)
df_raw_session.to_csv(raw_path + 'sessions_clean.csv', index=False)




