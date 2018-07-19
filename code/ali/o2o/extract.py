# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np



class extract(object):
    
    def __init__(self,data_path):
        self.df_offline = pd.read_csv(os.path.join(os.path.join(data_path, 'temp'), 'offline_part1.csv'),
                                  dtype=str, 
                                  keep_default_na=False)
        
        self.df_online = pd.read_csv(os.path.join(os.path.join(data_path, 'temp'), 'online_part1.csv'),
                                  dtype=str, 
                                  keep_default_na=False)
    
    def user_offline_features(self):
        # Total user number
        total_data_number1 = len(self.df_offline)
        print('Total offline user number is %d' % len(self.df_offline['User_id'].drop_duplicates()))
        
        # Feature 1
        df_feature1 = self.df_offline[['User_id', 'Coupon_id']]
        df_feature1['Coupon_id'] = df_feature1.Coupon_id.apply(lambda x : None if x == 'null' else x)
        df_feature1 = df_feature1.groupby(['User_id'], as_index=False)['Coupon_id'].count()
        df_feature1.rename(columns={'Coupon_id':'get_cp_num'}, inplace=True)
        
        # Feature 2
        df_feature2_1 = self.df_offline[(self.df_offline.Date_received != 'null') & (self.df_offline.Date == 'null')][['User_id', 'Date_received']]
        df_feature2_1 = df_feature2_1.groupby(['User_id'], as_index=False).count()
        df_feature2_1.rename(columns={'Date_received':'get_cp_not_use'}, inplace=True)
        
        df_feature2_2 = self.df_offline[(self.df_offline.Date_received != 'null') & (self.df_offline.Date != 'null')][['User_id', 'Date_received']]
        df_feature2_2.iloc[:,-1] = '0'
        df_feature2_2.rename(columns={'Date_received':'get_cp_not_use'}, inplace=True)
        
        df_feature2_3 = self.df_offline[(self.df_offline.Date_received == 'null')][['User_id', 'Date_received']]
        df_feature2_3.iloc[:,-1] = '0'
        df_feature2_3.rename(columns={'Date_received':'get_cp_not_use'}, inplace=True)
        
        df_feature2 = pd.concat([df_feature2_1, df_feature2_2, df_feature2_3])
        df_feature2.iloc[:,-1] = df_feature2.get_cp_not_use.astype('int')
        df_feature2 = df_feature2.groupby(['User_id'], as_index=False)['get_cp_not_use'].sum()

        # Feature 3
        df_feature3_1 = self.df_offline[(self.df_offline.Date_received != 'null') & (self.df_offline.Date != 'null')][['User_id', 'Date']]
        df_feature3_1 = df_feature3_1.groupby(['User_id'], as_index=False).count()
        df_feature3_1.rename(columns={'Date':'get_cp_used'}, inplace=True)
        
        df_feature3 = pd.DataFrame(self.df_offline['User_id'].drop_duplicates())
        df_feature3 = pd.merge(df_feature3, df_feature3_1, how='left', on=['User_id'])
        df_feature3.fillna(0, inplace=True)
        df_feature3.iloc[:,-1] = df_feature3.get_cp_used.astype('int')
        
        # Feature 4
        df_feature4_1 = self.df_offline.groupby(['User_id'], as_index=False)['Merchant_id'].count()
        df_feature4_1.rename(columns={'Merchant_id':'total'}, inplace=True)
        df_feature4_2 = self.df_offline[(self.df_offline.Date_received != 'null') & (self.df_offline.Date != 'null')][['User_id', 'Date']]
        df_feature4_2 = df_feature4_2.groupby(['User_id'], as_index=False).count()
        df_feature4_2.rename(columns={'Date':'used'}, inplace=True)
        df_feature4 = pd.merge(df_feature4_1, df_feature4_2, how='left', on=['User_id'])
        df_feature4.fillna(0, inplace=True)
        df_feature4.iloc[:,-2] = df_feature4.total.astype('float')
        df_feature4['used_rate'] = df_feature4.apply(lambda x:x[2]/x[1], axis=1)
        df_feature4.drop(['total','used'], axis=1, inplace=True)
        
        # Feature 5
        pass
        
        # Verify part to make sure there is no data lost
        total_data_number2 = len(self.df_offline)
        if total_data_number1 - total_data_number2 != 0:
            raise RuntimeError('Data lost')
        
        




if __name__ == '__main__':
    extr = extract('C:\\scnguh\\datamining\\o2o')
    extr.user_offline_features()
    
    
    
    
    
    
    
