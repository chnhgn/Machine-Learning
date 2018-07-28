# -*- coding: utf-8 -*-
import xgboost as xgb
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


class model(object):
    
    def __init__(self, data_path):
        self.feature_data_dir = os.path.join(data_path, 'tmp_features')     # Save feature data
        
        self.split_data_dir = os.path.join(data_path, 'temp')
        
        self.valid = pd.read_csv(os.path.join(data_path, 'ccf_offline_stage1_test_revised.csv'),
                                  dtype=str, 
                                  keep_default_na=False)
        
        self.train = pd.read_csv(os.path.join(self.feature_data_dir, 'o2o_train.csv'),
                            dtype=str,
                            keep_default_na=False)
        
        self.test = pd.read_csv(os.path.join(self.split_data_dir, 'offline_part2.csv'),
                            dtype=str,
                            keep_default_na=False)
    
    def modeling(self):
        # Processing the valid data
        self.valid['Distance'] = self.valid.Distance.apply(lambda x : '10' if x == 'null' else x)  # The null distance will be regarded as the farthest
        
        # Convert the discount to numeric value
        self.valid['Discount_rate'] = self.valid.Discount_rate.apply(lambda x:x if ':' not in x 
                                                else (float(str(x).split(':')[0]) - float(str(x).split(':')[1]))/float(str(x).split(':')[0]))
        
        # Save the valid data to feature directory
        self.valid.to_csv(os.path.join(self.feature_data_dir, 'o2o_valid.csv'), index=False)
        
        self.valid['Discount_rate'] = self.valid.Discount_rate.astype('float')
        self.valid['Distance'] = self.valid.Distance.astype('float')
        
        # Processing the test data
        self.test = self.test[(self.test.Coupon_id != 'null')]
        self.test['target'] = self.test.Date.apply(lambda x : '1' if x != 'null' else '0')
        self.test.drop(['Date'], axis=1, inplace=True)
        self.test['Distance'] = self.test.Distance.apply(lambda x : '10' if x == 'null' else x)
        self.test['Discount_rate'] = self.test.Discount_rate.apply(lambda x:x if ':' not in x 
                                    else (float(str(x).split(':')[0]) - float(str(x).split(':')[1]))/float(str(x).split(':')[0]))
        
        # Split the data and label for train/validate
        self.test['Discount_rate'] = self.test.Discount_rate.astype('float')
        self.test['Distance'] = self.test.Distance.astype('float')
        
        copy = self.test.copy()       
        test_y = np.array(copy['target'].tolist()).astype(np.int)
        self.test.drop(['target'], axis=1, inplace=True)
        test_x = self.test
        
        self.train.drop(['cp_received_dayofweek', 'cp_received_dayofmonth'], axis=1, inplace=True)
        for col in self.train.columns:
            if col not in ['User_id', 'Merchant_id', 'Coupon_id', 'target', 'cp_type']:
                self.train[col] = eval("self.train['%s'].astype('float')" % col)
        
        copy1 = self.train.copy()       
        train_y = np.array(copy1['target'].tolist()).astype(np.int)
        self.train.drop(['target'], axis=1, inplace=True)
        train_x = self.train
        
        # Need to encode string values to integer for xgboost
        for f in train_x.columns: 
            if train_x[f].dtype=='object': 
                lbl = preprocessing.LabelEncoder() 
                lbl.fit(list(train_x[f].values)) 
                train_x[f] = lbl.transform(list(train_x[f].values))
                
        for f in test_x.columns: 
            if test_x[f].dtype=='object': 
                lbl = preprocessing.LabelEncoder() 
                lbl.fit(list(test_x[f].values)) 
                test_x[f] = lbl.transform(list(test_x[f].values))
        
        df_tmp = train_x.copy().iloc[0:0]
        test_x = pd.concat([df_tmp, test_x])
#         test_x.fillna(-9999, inplace=True)
        
        train_x = train_x[sorted(train_x.columns)]
        test_x = test_x[sorted(test_x.columns)]
        
        # Build model
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test_x)
        
        params={'booster':'gbtree',
                'objective': 'rank:pairwise',
                'eval_metric':'auc',
                'gamma':0.1,
                'min_child_weight':1.1,
                'max_depth':5,
                'lambda':10,
                'subsample':0.7,
                'colsample_bytree':0.7,
                'colsample_bylevel':0.7,
                'eta': 0.01,
                'tree_method':'exact',
                'seed':0,
                'nthread':12
            }
        
        watchlist = [(dtrain,'train')]
        
        bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)

        ypred=bst.predict(dtest)
        
        print('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))




if __name__ == '__main__':
    
    mo = model('C:\\scnguh\\datamining\\o2o')
    
    mo.modeling()
