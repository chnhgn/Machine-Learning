# -*- coding: utf-8 -*-
import os
import pandas as pd
import scipy.stats as stats


class analysis(object):
    
    def __init__(self, data_path):
        self.feature_data_dir = os.path.join(data_path, 'tmp_features')
        
        self.train = pd.read_csv(os.path.join(self.feature_data_dir, 'o2o_train.csv'),
                            dtype=str,
                            keep_default_na=False)
    
    def correlation(self):
        columns = list(self.train.columns)
        for col in columns:
            # Non-string variables
            if col not in ['User_id', 'Merchant_id', 'Coupon_id', 'Date_received', 'target', 'cp_received_dayofweek', 'cp_received_dayofmonth']:
                self.train[col] = self.train[col].astype('float')
        
        df_corr = self.train.corr().abs()
        df_corr = df_corr[(df_corr > 0.7)]
#         print((df_corr.stack()[df_corr.stack() > 0.5]).unstack())
        
        df_corr.to_csv(os.path.join(self.feature_data_dir, 'o2o_corr.csv'))






if __name__ == '__main__':
    ana = analysis('C:\\scnguh\\datamining\\o2o')
    
    ana.correlation()
    
