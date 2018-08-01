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
        
        self.predict = pd.read_csv(os.path.join(self.feature_data_dir, 'o2o_predict.csv'),
                            dtype=str,
                            keep_default_na=False)
        
        self.predict_copy = self.predict.copy()
        
        self.train = pd.read_csv(os.path.join(self.feature_data_dir, 'o2o_train.csv'),
                            dtype=str,
                            keep_default_na=False)
        
        self.test = pd.read_csv(os.path.join(self.feature_data_dir, 'o2o_test.csv'),
                            dtype=str,
                            keep_default_na=False)
    
    def preprocessing(self):
        result = {}
        for data in ['self.train', 'self.test', 'self.predict']:
            
            eval("%s.drop(['User_id', 'Merchant_id', 'Coupon_id'], axis=1, inplace=True)" % data)
            for col in eval("%s.columns" % data):
                if col not in ['target', 'cp_type', 'Date_received', 'cp_received_dayofweek', 'cp_received_dayofmonth']:
                    exec("%s[col] = %s[col].astype('float')" % (data, data))
            
            copy = eval("%s.copy()" % data)
            if data != 'self.predict':
                label = np.array(copy['target'].tolist()).astype(np.int)
                copy.drop(['target'], axis=1, inplace=True)
            else:
                label = ''    
            
            for f in copy.columns: 
                if copy[f].dtype=='object': 
                    lbl = preprocessing.LabelEncoder() 
                    lbl.fit(list(copy[f].values)) 
                    copy[f] = lbl.transform(list(copy[f].values))
            
            name = data.split('.')[1]
            x = name + '_x'
            y = name + '_y'
            
            result[x] = copy
            result[y] = label
        
        return result
    
    def modeling(self, train, label, test_data=None, test_label=None, predict=None):
        # Build model
        dtrain = xgb.DMatrix(train, label=label)
        
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
        
        bst=xgb.train(params,dtrain,num_boost_round=1500,evals=watchlist)
        
        if predict is not None:
            dpredict = xgb.DMatrix(predict)
            y_predict = bst.predict(dpredict)
            
            # Persist the predict result
            df = self.predict_copy
            df = df[['User_id', 'Coupon_id', 'Date_received']]
            df_prob = pd.DataFrame(y_predict, columns=['Probability'])
            df = pd.concat([df, df_prob], axis=1)
            df.to_csv(os.path.join(self.feature_data_dir, 'o2o_predict_result.csv'), index=False)
        
        if test_data is not None and test_label is not None:
            dtest = xgb.DMatrix(test_data)
            ypred=bst.predict(dtest)
            print('AUC: %.4f' % metrics.roc_auc_score(test_label,ypred))
        
        



if __name__ == '__main__':
    
    mo = model('C:\\scnguh\\datamining\\o2o')
    
    res = mo.preprocessing()
    
#     mo.modeling(res['train_x'], res['train_y'], res['test_x'], res['test_y'])
    
    mo.modeling(res['train_x'], res['train_y'], predict=res['predict_x'])
    

    
    
    
    
    
    
    
