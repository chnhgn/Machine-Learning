# -*- coding: utf-8 -*-
import xgboost as xgb
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


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
        
        self.test_copy = self.test.copy()
    
    def preprocessing(self):
        result = {}
        for data in ['self.train', 'self.test', 'self.predict']:
            
            eval("%s.drop(['User_id', 'Merchant_id', 'Coupon_id'], axis=1, inplace=True)" % data)
            for col in eval("%s.columns" % data):
                if col not in ['target', 'cp_type', 'Date_received', 'cp_received_dayofweek', 'cp_received_dayofmonth']:
                    exec("%s[col] = %s[col].astype('float')" % (data, data))
            
            copy = eval("%s.copy()" % data)
            copy.sort_index(axis=1, inplace=True)
            
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
            df_prob = pd.DataFrame(MinMaxScaler().fit_transform(y_predict.reshape(-1, 1)), columns=['Probability'])
            df = pd.concat([df, df_prob], axis=1)
            df.to_csv(os.path.join(self.feature_data_dir, 'o2o_predict_result.csv'), index=False)
        
        if test_data is not None and test_label is not None:
            dtest = xgb.DMatrix(test_data)
            ypred=bst.predict(dtest)
            
            df2 = self.test_copy
            df2 = df2[['User_id', 'Coupon_id', 'Date_received', 'target']]
            df_prob2 = pd.DataFrame(MinMaxScaler().fit_transform(ypred.reshape(-1, 1)), columns=['Probability'])
            df2 = pd.concat([df2, df_prob2], axis=1)
            print('AUC: %.4f' % metrics.roc_auc_score(test_label,ypred))
            df2.to_csv(os.path.join(self.feature_data_dir, 'o2o_predict_test.csv'), index=False)
            
    def test_model(self):
        train_x, test_x, train_y, test_y = train_test_split(res['train_x'], res['train_y'], random_state=0) 
        
        dtrain=xgb.DMatrix(train_x,label=train_y)
        dtest=xgb.DMatrix(test_x)
        
        params={'booster':'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth':4,
            'lambda':10,
            'subsample':0.75,
            'colsample_bytree':0.75,
            'min_child_weight':2,
            'eta': 0.025,
            'seed':0,
            'nthread':8,
             'silent':1}
        
        watchlist = [(dtrain,'train')]   
        bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)
        ypred=bst.predict(dtest)
        
        df_prob = pd.DataFrame(MinMaxScaler().fit_transform(ypred.reshape(-1, 1)), columns=['Probability'])
        df_label = pd.DataFrame(test_y, columns=['Label'])
        df = pd.concat([df_prob, df_label], axis=1)
        df.to_csv(os.path.join(self.feature_data_dir, 'o2o_predict_test.csv'), index=False)
        print('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))


if __name__ == '__main__':
    
    mo = model('C:\\scnguh\\datamining\\o2o')
    
    res = mo.preprocessing()
    
#     mo.test_model()
    
#     mo.modeling(res['train_x'], res['train_y'], res['test_x'], res['test_y'])
    
    mo.modeling(res['train_x'], res['train_y'], predict=res['predict_x'])
    

    
    
    
    
    
    
    
