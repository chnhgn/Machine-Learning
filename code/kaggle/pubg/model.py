# -*- coding: utf-8 -*-
'''
Created on Nov 15, 2018

@author: Eddy Hu
'''

import pandas as pd
import numpy as np
import xgboost as xgb


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
test_id = test.Id.values
train.drop(['Id', 'groupId', 'matchId'], 1, inplace=True)
test.drop(['Id', 'groupId', 'matchId'], 1, inplace=True)

df_all = pd.concat([train, test])

for feature in ['matchType']:
    dummy_features = pd.get_dummies(df_all[feature], prefix=feature)
    for dummy in dummy_features:
        df_all[dummy] = dummy_features[dummy]
    df_all.drop([feature], 1, inplace=True)
    
for index, value in df_all.dtypes.iteritems():
    minValue = df_all[index].min()
    maxValue = df_all[index].max()
    df_all[index] = df_all[index].apply(lambda x:(x - minValue) / (maxValue - minValue))    
    
train = df_all[:-len(test)]
test = df_all[-len(test):] 

yTrain = np.array(train.winPlacePerc)
train.drop(['winPlacePerc'], 1, inplace=True)
test.drop(['winPlacePerc'], 1, inplace=True)

num_trees = 100
params = {"objective": "reg:linear",
          "eta": 0.15,
          "max_depth": 8,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 1
          }

dtrain = xgb.DMatrix(train, label=yTrain)
dtest = xgb.DMatrix(test)
watchlist = [(dtrain, 'train')]

gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, verbose_eval=True)

yhat = gbm.predict(dtest)

arr = np.vstack((test_id, yhat))
arr = arr.T
df_final = pd.DataFrame(arr, columns=['Id', 'winPlacePerc'])
df_final.to_csv("./data/submission1.csv", index=False)
print("Generated submission file!")
