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

        # [Feature 11]
        df_feature11_1 = self.df_offline[(self.df_offline.Discount_rate != 'null') & (self.df_offline.Date != 'null')][['User_id', 'Distance']]
        df_feature11_1.iloc[:,-1] = df_feature11_1.Distance.apply(lambda x:'10' if x == 'null' else x)
        df_feature11_1.iloc[:,-1] = df_feature11_1.Distance.astype('float')
        df_feature11_2 = df_feature11_1.groupby(['User_id'], as_index=False)['Distance'].max()
        df_feature11_2.rename(columns={'Distance':'distance_max'}, inplace=True)
        df_feature11_3 = df_feature11_1.groupby(['User_id'], as_index=False)['Distance'].min()
        df_feature11_3.rename(columns={'Distance':'distance_min'}, inplace=True)
        df_feature11_4 = df_feature11_1.groupby(['User_id'], as_index=False)['Distance'].mean()
        df_feature11_4.rename(columns={'Distance':'distance_mean'}, inplace=True)
        
        df_feature11_5 = pd.merge(pd.merge(df_feature11_2, df_feature11_3, on=['User_id']), df_feature11_4, on=['User_id'])
        
        df_feature11_6 = self.df_offline[['User_id']]
        df_feature11_6.drop_duplicates(inplace=True)
        df_feature11 = pd.merge(df_feature11_6, df_feature11_5, how='left', on=['User_id'])
        df_feature11.fillna(0, inplace=True)
        
        # [Feature 10]
        df_feature10_1 = self.df_offline[(self.df_offline.Discount_rate != 'null') & (self.df_offline.Date != 'null')][['User_id', 'Coupon_id']]
        df_feature10_1 = df_feature10_1.groupby(['User_id'], as_index=False)['Coupon_id'].count()
        df_feature10_1.iloc[:,-1] = df_feature10_1.Coupon_id.astype('float')
        
        df_feature10_2 = self.df_offline[(self.df_offline.Discount_rate != 'null') & (self.df_offline.Date != 'null')][['User_id', 'Merchant_id', 'Date']]
        df_feature10_2 = df_feature10_2.groupby(['User_id', 'Merchant_id'], as_index=False)['Date'].count()
        df_feature10_2 = df_feature10_2.groupby(['User_id'], as_index=False)['Merchant_id'].count()
        df_feature10_2.iloc[:,-1] = df_feature10_2.Merchant_id.astype('float')
          
        df_feature10_3 = pd.merge(df_feature10_1, df_feature10_2, on=['User_id'])
        df_feature10_3['cp_used_per_shop'] = df_feature10_3.apply(lambda x:x[-2]/x[-1], axis=1)
        df_feature10_3.drop(['Coupon_id', 'Merchant_id'], axis=1, inplace=True)
        
        df_feature10_4 = self.df_offline[['User_id']]
        df_feature10_4.drop_duplicates(inplace=True)
        
        df_feature10 = pd.merge(df_feature10_4, df_feature10_3, how='left', on=['User_id'])
        df_feature10.fillna(0, inplace=True)

        # [Feature 9]
        df_feature9_1 = self.df_offline[['Coupon_id']]
        df_feature9_1.drop_duplicates(inplace=True)
        cp_num = len(df_feature9_1)
        
        df_feature9_2 = self.df_offline[(self.df_offline.Discount_rate != 'null') & (self.df_offline.Date != 'null')][['User_id', 'Coupon_id']]
        df_feature9_2 = df_feature9_2.groupby(['User_id'], as_index=False)['Coupon_id'].count()
        df_feature9_2.iloc[:,-1] = df_feature9_2.Coupon_id.astype('float')
        df_feature9_2.iloc[:,-1] = df_feature9_2.Coupon_id.apply(lambda x:x/cp_num)
        
        df_feature9_3 = self.df_offline[['User_id']]
        df_feature9_3.drop_duplicates(inplace=True)
        df_feature9 = pd.merge(df_feature9_3, df_feature9_2, how='left', on=['User_id'])
        df_feature9.fillna(0, inplace=True)
        
        # [Feature 8]
        df_feature8_1 = self.df_offline[['Merchant_id']]
        df_feature8_1.drop_duplicates(inplace=True)
        shop_num = len(df_feature8_1)
        
        df_feature8_2 = self.df_offline[(self.df_offline.Discount_rate != 'null') & (self.df_offline.Date != 'null')][['User_id', 'Merchant_id']]
        df_feature8_2 = df_feature8_2.groupby(['User_id'], as_index=False)['Merchant_id'].count()
        df_feature8_2.iloc[:,-1] = df_feature8_2.Merchant_id.astype('float')
        df_feature8_2.iloc[:,-1] = df_feature8_2.Merchant_id.apply(lambda x:x/shop_num)
        
        df_feature8_3 = self.df_offline[['User_id']]
        df_feature8_3.drop_duplicates(inplace=True)
        df_feature8 = pd.merge(df_feature8_3, df_feature8_2, how='left', on=['User_id'])
        df_feature8.fillna(0, inplace=True)
        
        # [Feature 7]
        df_feature7_1 = self.df_offline[(self.df_offline.Discount_rate != 'null') & (self.df_offline.Date != 'null')][['User_id', 'Discount_rate']]
        df_feature7_1.iloc[:,-1] = df_feature7_1.Discount_rate.apply(lambda x:x if ':' not in x 
                                                                     else (float(str(x).split(':')[0]) - float(str(x).split(':')[1]))/float(str(x).split(':')[0]))
        df_feature7_1.iloc[:,-1] = df_feature7_1.Discount_rate.astype('float')
        df_feature7_2 = df_feature7_1.groupby(['User_id'], as_index=False)['Discount_rate'].mean()
        df_feature7_2.rename(columns={'Discount_rate':'cp_used_discount_mean'}, inplace=True)
        df_feature7_3 = df_feature7_1.groupby(['User_id'], as_index=False)['Discount_rate'].max()
        df_feature7_3.rename(columns={'Discount_rate':'cp_used_discount_max'}, inplace=True)
        df_feature7_4 = df_feature7_1.groupby(['User_id'], as_index=False)['Discount_rate'].min()
        df_feature7_4.rename(columns={'Discount_rate':'cp_used_discount_min'}, inplace=True)
        
        df_feature7 = pd.merge(df_feature7_2, df_feature7_3, on=['User_id'])
        df_feature7 = pd.merge(df_feature7, df_feature7_4, on=['User_id'])
        
        df_feature7_5 = self.df_offline[['User_id']]
        df_feature7_5.drop_duplicates(inplace=True)
        df_feature7 = pd.merge(df_feature7_5, df_feature7, how='left', on=['User_id'])
        df_feature7.fillna(0, inplace=True)
        
        # [Feature 6]
        df_feature6_1 = self.df_offline[(self.df_offline.Discount_rate != 'null')][['User_id', 'Discount_rate', 'Date']]
        # Discount:0 Coupon:1
        df_feature6_1['Discount_rate'] = df_feature6_1.Discount_rate.apply(lambda x:'0' if '.' in x else ('1' if ':' in x else x))
        df_feature6_1['Date'] = df_feature6_1.Date.apply(lambda x:None if x == 'null' else x)
        df_feature6_1 = df_feature6_1.groupby(['User_id', 'Discount_rate'], as_index=False)['Date'].count()
        df_feature6_1.rename(columns={'Discount_rate':'cp_type', 'Date':'used_num'}, inplace=True)
        df_feature6_1 = df_feature6_1[df_feature6_1.cp_type == '1']
        df_feature6_1.drop(['cp_type'], axis=1, inplace=True)
        df_feature6_1 = df_feature6_1.groupby(['User_id'], as_index=False)['used_num'].sum()
        df_feature6_1.rename(columns={'used_num':'x:y_cp_used_num'}, inplace=True)
        
        # all used coupon
        df_feature6_2 = self.df_offline[(self.df_offline.Date != 'null')][['User_id', 'Date']]
        df_feature6_2 = df_feature6_2.groupby(['User_id'], as_index=False)['Date'].count()
        df_feature6_2.rename(columns={'Date':'all_cp_used_num'}, inplace=True)
        
        df_feature6_3 = self.df_offline[['User_id']]
        df_feature6_3.drop_duplicates(inplace=True)
        df_feature6 = pd.merge(df_feature6_3, df_feature6_1, how='left', on=['User_id'])
        df_feature6 = pd.merge(df_feature6, df_feature6_2, how='left', on=['User_id'])
        df_feature6.fillna(0, inplace=True)
        df_feature6['x:y_in_all_cp_used_rate'] = df_feature6.apply(lambda x:x[1]/x[2] if x[2] != 0 else 0, axis=1)
        df_feature6 = df_feature6[['User_id', 'x:y_in_all_cp_used_rate']]
        
        # [Feature 5]
        df_feature5_1 = self.df_offline[(self.df_offline.Discount_rate != 'null')][['User_id', 'Discount_rate', 'Date']]
        # Discount:0 Coupon:1
        df_feature5_1['Discount_rate'] = df_feature5_1.Discount_rate.apply(lambda x:'0' if '.' in x else ('1' if ':' in x else x))
        df_feature5_1['Date'] = df_feature5_1.Date.apply(lambda x:None if x == 'null' else x)
        df_feature5_1 = df_feature5_1.groupby(['User_id', 'Discount_rate'], as_index=False)['Date'].count()
        df_feature5_1.rename(columns={'Discount_rate':'cp_type', 'Date':'used_num'}, inplace=True)
        df_feature5_1 = df_feature5_1[df_feature5_1.cp_type == '1']
        
        # All types coupon
        df_feature5_2 = self.df_offline[['User_id', 'Coupon_id']]
        df_feature5_2['Coupon_id'] = df_feature5_2.Coupon_id.apply(lambda x:None if x == 'null' else x)
        df_feature5_2 = df_feature5_2.groupby(['User_id'], as_index=False)['Coupon_id'].count()
        df_feature5_2.rename(columns={'Coupon_id':'all_cp_num'}, inplace=True)
        df_feature5 = pd.merge(df_feature5_2, df_feature5_1, how='left', on=['User_id'])
        df_feature5.drop(['cp_type'], axis=1, inplace=True)
        df_feature5.fillna(0, inplace=True)
        df_feature5.iloc[:,-2] = df_feature5.all_cp_num.astype('float')
        df_feature5['x:y_cp_used_rate'] = df_feature5.apply(lambda x:x[-1]/x[-2] if x[-2] != 0 else 0, axis=1)
        df_feature5.drop(df_feature5.columns[[1,2]], axis=1, inplace=True)
        
        # [Feature 4]
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
        
        # [Feature 3]
        df_feature3_1 = self.df_offline[(self.df_offline.Date_received != 'null') & (self.df_offline.Date != 'null')][['User_id', 'Date']]
        df_feature3_1 = df_feature3_1.groupby(['User_id'], as_index=False).count()
        df_feature3_1.rename(columns={'Date':'get_cp_used'}, inplace=True)
        
        df_feature3 = pd.DataFrame(self.df_offline['User_id'].drop_duplicates())
        df_feature3 = pd.merge(df_feature3, df_feature3_1, how='left', on=['User_id'])
        df_feature3.fillna(0, inplace=True)
        df_feature3.iloc[:,-1] = df_feature3.get_cp_used.astype('int')
        
        # [Feature 2]
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
        
        # [Feature 1]
        df_feature1 = self.df_offline[['User_id', 'Coupon_id']]
        df_feature1['Coupon_id'] = df_feature1.Coupon_id.apply(lambda x : None if x == 'null' else x)
        df_feature1 = df_feature1.groupby(['User_id'], as_index=False)['Coupon_id'].count()
        df_feature1.rename(columns={'Coupon_id':'get_cp_num'}, inplace=True)
        
        # Verify part to make sure there is no data lost
        total_data_number2 = len(self.df_offline)
        if total_data_number1 - total_data_number2 != 0:
            raise RuntimeError('Data lost')
        
        




if __name__ == '__main__':
    extr = extract('C:\\scnguh\\datamining\\o2o')
    extr.user_offline_features()
    
    
    
    
    
    
    
