# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np



class extract(object):
    
    def __init__(self,data_path):
        self.split_data_dir = os.path.join(data_path, 'temp')   # Save split data
        
        self.feature_data_dir = os.path.join(data_path, 'tmp_features')     # Save feature data
        if os.path.exists(self.feature_data_dir) is not True:
            os.mkdir(self.feature_data_dir)
        
        self.df_offline = pd.read_csv(os.path.join(self.split_data_dir, 'offline_part1.csv'),
                                  dtype=str, 
                                  keep_default_na=False)
        
        self.df_online = pd.read_csv(os.path.join(self.split_data_dir, 'online_part1.csv'),
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
        df_feature11.set_index(['User_id'], inplace=True)     # Merge use
        
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
        df_feature10.set_index(['User_id'], inplace=True)     # Merge use

        # [Feature 9]
        df_feature9_1 = self.df_offline[['Coupon_id']]
        df_feature9_1.drop_duplicates(inplace=True)
        cp_num = len(df_feature9_1)
        
        df_feature9_2 = self.df_offline[(self.df_offline.Discount_rate != 'null') & (self.df_offline.Date != 'null')][['User_id', 'Coupon_id']]
        df_feature9_2 = df_feature9_2.groupby(['User_id'], as_index=False)['Coupon_id'].count()
        df_feature9_2.iloc[:,-1] = df_feature9_2.Coupon_id.astype('float')
        df_feature9_2.iloc[:,-1] = df_feature9_2.Coupon_id.apply(lambda x:x/cp_num)
        df_feature9_2.rename(columns={'Coupon_id':'get_cp_used_rate'}, inplace=True)
        
        df_feature9_3 = self.df_offline[['User_id']]
        df_feature9_3.drop_duplicates(inplace=True)
        df_feature9 = pd.merge(df_feature9_3, df_feature9_2, how='left', on=['User_id'])
        df_feature9.fillna(0, inplace=True)
        df_feature9.set_index(['User_id'], inplace=True)     # Merge use
        
        # [Feature 8]
        df_feature8_1 = self.df_offline[['Merchant_id']]
        df_feature8_1.drop_duplicates(inplace=True)
        shop_num = len(df_feature8_1)
        
        df_feature8_2 = self.df_offline[(self.df_offline.Discount_rate != 'null') & (self.df_offline.Date != 'null')][['User_id', 'Merchant_id']]
        df_feature8_2 = df_feature8_2.groupby(['User_id'], as_index=False)['Merchant_id'].count()
        df_feature8_2.iloc[:,-1] = df_feature8_2.Merchant_id.astype('float')
        df_feature8_2.iloc[:,-1] = df_feature8_2.Merchant_id.apply(lambda x:x/shop_num)
        df_feature8_2.rename(columns={'Merchant_id':'cp_used_merchants_outof_all_merchants'}, inplace=True)
        
        df_feature8_3 = self.df_offline[['User_id']]
        df_feature8_3.drop_duplicates(inplace=True)
        df_feature8 = pd.merge(df_feature8_3, df_feature8_2, how='left', on=['User_id'])
        df_feature8.fillna(0, inplace=True)
        df_feature8.set_index(['User_id'], inplace=True)     # Merge use
        
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
        df_feature7.set_index(['User_id'], inplace=True)     # Merge use
        
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
        df_feature6.set_index(['User_id'], inplace=True)     # Merge use
        
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
        df_feature5.set_index(['User_id'], inplace=True)     # Merge use
        
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
        df_feature4.set_index(['User_id'], inplace=True)     # Merge use
        
        # [Feature 3]
        df_feature3_1 = self.df_offline[(self.df_offline.Date_received != 'null') & (self.df_offline.Date != 'null')][['User_id', 'Date']]
        df_feature3_1 = df_feature3_1.groupby(['User_id'], as_index=False).count()
        df_feature3_1.rename(columns={'Date':'get_cp_used'}, inplace=True)
        
        df_feature3 = pd.DataFrame(self.df_offline['User_id'].drop_duplicates())
        df_feature3 = pd.merge(df_feature3, df_feature3_1, how='left', on=['User_id'])
        df_feature3.fillna(0, inplace=True)
        df_feature3.iloc[:,-1] = df_feature3.get_cp_used.astype('int')
        df_feature3.set_index(['User_id'], inplace=True)     # Merge use
        
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
        df_feature2.set_index(['User_id'], inplace=True)     # Merge use
        
        # [Feature 1]
        df_feature1 = self.df_offline[['User_id', 'Coupon_id']]
        df_feature1['Coupon_id'] = df_feature1.Coupon_id.apply(lambda x : None if x == 'null' else x)
        df_feature1 = df_feature1.groupby(['User_id'], as_index=False)['Coupon_id'].count()
        df_feature1.rename(columns={'Coupon_id':'get_cp_num'}, inplace=True)
        df_feature1.set_index(['User_id'], inplace=True)     # Merge use
        
        # Merge all single features
        df_ft = pd.concat([df_feature1, df_feature2, df_feature3, df_feature4, df_feature5, df_feature6,
                            df_feature7, df_feature8, df_feature9, df_feature10, df_feature11], axis=1)
        df_ft.reset_index(inplace=True)
        df_ft.rename(columns={'index':'User_id'}, inplace=True)
        df_ft.to_csv(os.path.join(self.feature_data_dir, 'user_offline_features.csv'), index=False)
        
        # Verify part to make sure there is no data lost
        total_data_number2 = len(self.df_offline)
        if total_data_number1 - total_data_number2 != 0:
            raise RuntimeError('Data lost')
        
    def user_online_features(self):
        # Total user number
        print('Total online user number is %d' % len(self.df_online['User_id'].drop_duplicates()))

        # [feature 10]
        df_ft8_1 = self.df_offline[(self.df_offline.Date_received != 'null')][['User_id', 'Date_received']]
        df_ft8_1 = df_ft8_1.groupby(['User_id'], as_index=False)['Date_received'].count()
        df_ft8_1.rename(columns={'Date_received':'offline_cp_record'}, inplace=True)
        
        df_ft8_2 = self.df_online[(self.df_online.Date_received != 'null')][['User_id', 'Date_received']]
        df_ft8_2 = df_ft8_2.groupby(['User_id'], as_index=False)['Date_received'].count()
        df_ft8_2.rename(columns={'Date_received':'online_cp_record'}, inplace=True)
        
        df_ft8_3 = self.df_online[['User_id']].drop_duplicates()
        df_ft8 = pd.merge(pd.merge(df_ft8_3, df_ft8_2, how='left', on=['User_id']), df_ft8_1, how='left', on=['User_id'])
        df_ft8.fillna(0, inplace=True)
        df_ft8['offline_cp_record_in_online_and_offline'] = df_ft8.apply(lambda x:x[1]/(x[1]+x[2]) if x[1]+x[2] != 0 else 0, axis=1)
        df_ft8 = df_ft8[['User_id', 'offline_cp_record_in_online_and_offline']]
        df_ft8.set_index(['User_id'], inplace=True)     # Merge use
        
        # [feature 9]
        df_ft7_1 = self.df_offline[(self.df_offline.Date != 'null') & (self.df_offline.Coupon_id != 'null')][['User_id', 'Date']]
        df_ft7_1 = df_ft7_1.groupby(['User_id'], as_index=False)['Date'].count()
        df_ft7_1.rename(columns={'Date':'offline_cp_used_num'}, inplace=True)
        
        df_ft7_2 = self.df_online[(self.df_online.Date != 'null') & (self.df_online.Coupon_id != 'null')][['User_id', 'Date']]
        df_ft7_2 = df_ft7_2.groupby(['User_id'], as_index=False)['Date'].count()
        df_ft7_2.rename(columns={'Date':'online_cp_used_num'}, inplace=True)
        
        df_ft7_3 = self.df_online[['User_id']].drop_duplicates()
        df_ft7 = pd.merge(pd.merge(df_ft7_3, df_ft7_2, how='left', on=['User_id']), df_ft7_1, how='left', on=['User_id'])
        df_ft7.fillna(0, inplace=True)
        df_ft7['offline_cp_used_num_in_online_and_offline'] = df_ft7.apply(lambda x:x[1]/(x[1]+x[2]) if x[1]+x[2] != 0 else 0, axis=1)
        df_ft7 = df_ft7[['User_id', 'offline_cp_used_num_in_online_and_offline']]
        df_ft7.set_index(['User_id'], inplace=True)     # Merge use
        
        # [feature 8]
        df_ft6_1 = self.df_offline[(self.df_offline.Date == 'null')][['User_id', 'Date']]
        df_ft6_1 = df_ft6_1.groupby(['User_id'], as_index=False)['Date'].count()
        df_ft6_1.rename(columns={'Date':'offline_no_consume_num'}, inplace=True)
        
        df_ft6_2 = self.df_online[(self.df_online.Date == 'null')][['User_id', 'Date']]
        df_ft6_2 = df_ft6_2.groupby(['User_id'], as_index=False)['Date'].count()
        df_ft6_2.rename(columns={'Date':'online_no_consume_num'}, inplace=True)
        
        df_ft6_3 = self.df_online[['User_id']].drop_duplicates()
        df_ft6 = pd.merge(pd.merge(df_ft6_3, df_ft6_2, how='left', on=['User_id']), df_ft6_1, how='left', on=['User_id'])
        df_ft6.fillna(0, inplace=True)
        df_ft6['offline_no_consume_num_in_online_and_offline'] = df_ft6.apply(lambda x:x[1]/(x[1]+x[2]) if x[1]+x[2] != 0 else 0, axis=1)
        df_ft6 = df_ft6[['User_id', 'offline_no_consume_num_in_online_and_offline']]
        df_ft6.set_index(['User_id'], inplace=True)     # Merge use
        
        
        # [feature 7]
        df_ft5_1 = self.df_online[['User_id']].drop_duplicates()
        
        df_ft5_2 = self.df_online[(self.df_online.Coupon_id != 'null')][['User_id', 'Coupon_id']]
        df_ft5_2 = df_ft5_2.groupby(['User_id'], as_index=False)['Coupon_id'].count()
        
        df_ft5_3 = self.df_online[(self.df_online.Coupon_id != 'null') & (self.df_online.Date != 'null')][['User_id', 'Date']]
        df_ft5_3 = df_ft5_3.groupby(['User_id'], as_index=False)['Date'].count()
        
        df_ft5_4 = pd.merge(df_ft5_2, df_ft5_3, how='left', on=['User_id'])
        df_ft5 = pd.merge(df_ft5_1, df_ft5_4, how='left', on=['User_id'])
        df_ft5.fillna(0, inplace=True)
        df_ft5['Coupon_id'] = df_ft5.Coupon_id.astype('float')
        df_ft5['Date'] = df_ft5.Date.astype('float')
        df_ft5['online_cp_used_rate'] = df_ft5.apply(lambda x:x[-1]/x[-2] if x[-2] != 0 else 0, axis=1)
        df_ft5 = df_ft5[['User_id', 'online_cp_used_rate']]
        df_ft5.set_index(['User_id'], inplace=True)     # Merge use
        
        # [feature 6]
        df_ft4_1 = self.df_online[(self.df_online.Coupon_id != 'null') & (self.df_online.Date != 'null')][['User_id', 'Date']]
        df_ft4_1 = df_ft4_1.groupby(['User_id'], as_index=False)['Date'].count()
        df_ft4_1.rename(columns={'Date':'online_cp_used_num'}, inplace=True)
        
        df_ft4_2 = self.df_online[['User_id']].drop_duplicates()
        
        df_ft4 = pd.merge(df_ft4_2, df_ft4_1, how='left', on=['User_id'])
        df_ft4.fillna(0, inplace=True)
        df_ft4.set_index(['User_id'], inplace=True)     # Merge use
        
        # [feature 5]
        df_ft3_1 = self.df_online[(self.df_online.Action.isin(['0', '2']))][['User_id', 'Action']]
        df_ft3_1 = df_ft3_1.groupby(['User_id'], as_index=False)['Action'].count()
        df_ft3_1.rename(columns={'Action':'online_no_consume_num'}, inplace=True)
        
        df_ft3_2 = self.df_online[['User_id']].drop_duplicates()
        df_ft3 = pd.merge(df_ft3_2, df_ft3_1, how='left', on=['User_id'])
        df_ft3.fillna(0, inplace=True)
        df_ft3.set_index(['User_id'], inplace=True)     # Merge use
        
        # [feature 2,3,4]
        df_ft2_1 = self.df_online[['User_id', 'Action']]
        df_ft2_1 = df_ft2_1.groupby(['User_id'], as_index=False)['Action'].count()
        df_ft2_1.rename(columns={'Action':'total_action_num'}, inplace=True)
        df_ft2_1.set_index(['User_id'], inplace=True)
        
        df_ft2_2 = self.df_online[(self.df_online.Action == '0')][['User_id', 'Action']]
        df_ft2_2 = df_ft2_2.groupby(['User_id'], as_index=False)['Action'].count()
        df_ft2_2.rename(columns={'Action':'hit_action_num'}, inplace=True)
        df_ft2_2.set_index(['User_id'], inplace=True)
        
        df_ft2_3 = self.df_online[(self.df_online.Action == '1')][['User_id', 'Action']]
        df_ft2_3 = df_ft2_3.groupby(['User_id'], as_index=False)['Action'].count()
        df_ft2_3.rename(columns={'Action':'buy_action_num'}, inplace=True)
        df_ft2_3.set_index(['User_id'], inplace=True)
        
        df_ft2_4 = self.df_online[(self.df_online.Action == '2')][['User_id', 'Action']]
        df_ft2_4 = df_ft2_4.groupby(['User_id'], as_index=False)['Action'].count()
        df_ft2_4.rename(columns={'Action':'get_cp_action_num'}, inplace=True)
        df_ft2_4.set_index(['User_id'], inplace=True)
        
        df_ft2 = pd.concat([df_ft2_1, df_ft2_2, df_ft2_3, df_ft2_4], axis=1)
        df_ft2.reset_index(inplace=True)
        df_ft2.rename(columns={'index':'User_id'}, inplace=True)
        df_ft2.fillna(0, inplace=True)
        df_ft2['hit_action_num'] = df_ft2.hit_action_num.astype('float')
        df_ft2['buy_action_num'] = df_ft2.buy_action_num.astype('float')
        df_ft2['get_cp_action_num'] = df_ft2.get_cp_action_num.astype('float')
        df_ft2['total_action_num'] = df_ft2.total_action_num.astype('float')
        df_ft2['online_hit_action_rate'] = df_ft2.apply(lambda x:x[2]/x[1] if x[1] != 0 else 0, axis=1)
        df_ft2['online_buy_action_rate'] = df_ft2.apply(lambda x:x[3]/x[1] if x[1] != 0 else 0, axis=1)
        df_ft2['online_get_cp_action_rate'] = df_ft2.apply(lambda x:x[4]/x[1] if x[1] != 0 else 0, axis=1)
        df_ft2 = df_ft2[['User_id', 'online_hit_action_rate', 'online_buy_action_rate', 'online_get_cp_action_rate']]
        df_ft2.set_index(['User_id'], inplace=True)     # Merge use
        
        # [Feature 1]
        df_ft1_1 = self.df_online[['User_id', 'Action']]
        df_ft1 = df_ft1_1.groupby(['User_id'], as_index=False)['Action'].count()
        df_ft1.rename(columns={'Action':'online_action_times'}, inplace=True)
        df_ft1.set_index(['User_id'], inplace=True)     # Merge use
        
        # Merge all single features
        df_ft = pd.concat([df_ft1, df_ft2, df_ft3, df_ft4, df_ft5, df_ft6, df_ft7, df_ft8], axis=1)
        df_ft.reset_index(inplace=True)
        df_ft.rename(columns={'index':'User_id'}, inplace=True)
        df_ft.to_csv(os.path.join(self.feature_data_dir, 'user_online_features.csv'), index=False)
        
    def merchant_features(self):
        # All merchants
        df_merchant = self.df_offline[['Merchant_id']].drop_duplicates()

        # [feature 10]
        df_ft10_1 = self.df_offline[(self.df_offline.Discount_rate != 'null') & (self.df_offline.Date != 'null')][['Merchant_id', 'Distance']]
        df_ft10_1['Distance'] = df_ft10_1.Distance.apply(lambda x : '10' if x == 'null' else x)
        df_ft10_1['Distance'] = df_ft10_1.Distance.astype('float')
        df_ft10_2 = df_ft10_1.groupby(['Merchant_id'], as_index=False)['Distance'].max()
        df_ft10_2.rename(columns={'Distance':'cp_used_max_distance_in_merchant'}, inplace=True)
        
        df_ft10_3 = df_ft10_1.groupby(['Merchant_id'], as_index=False)['Distance'].min()
        df_ft10_3.rename(columns={'Distance':'cp_used_min_distance_in_merchant'}, inplace=True)
        
        df_ft10_4 = df_ft10_1.groupby(['Merchant_id'], as_index=False)['Distance'].mean()
        df_ft10_4.rename(columns={'Distance':'cp_used_mean_distance_in_merchant'}, inplace=True)
        
        df_ft10 = pd.merge(pd.merge(df_ft10_2, df_ft10_3), df_ft10_4)
        df_ft10 = pd.merge(df_merchant, df_ft10, how='left', on=['Merchant_id'])
        df_ft10.fillna(10, inplace=True)
        
        df_ft10.set_index(['Merchant_id'], inplace=True)     # Merge use

        # [feature 9]
        df_ft9_1 = self.df_offline[(self.df_offline.Discount_rate != 'null') & (self.df_offline.Date != 'null')][['Merchant_id', 'Date_received', 'Date']]
        df_ft9_1['Date_received'] = pd.to_datetime(df_ft9_1.Date_received)
        df_ft9_1['Date'] = pd.to_datetime(df_ft9_1.Date)
        df_ft9_1['cp_used_duration'] = df_ft9_1.apply(lambda x : x[2] - x[1], axis=1)
        df_ft9_1['cp_used_duration'] = df_ft9_1.cp_used_duration.apply(lambda x: float(x.days))     # Convert timedelta to float
        df_ft9_1 = df_ft9_1[['Merchant_id', 'cp_used_duration']]
        df_ft9_1 = df_ft9_1.groupby(['Merchant_id'], as_index=False)['cp_used_duration'].mean()
        
        df_ft9 = pd.merge(df_merchant, df_ft9_1, how='left', on=['Merchant_id'])
        df_ft9.fillna(0, inplace=True)
        df_ft9.rename(columns={'cp_used_duration':'cp_used_duration_in_merchant'}, inplace=True)
        
        df_ft9.set_index(['Merchant_id'], inplace=True)     # Merge use
        
        # [feature 6]
        df_ft6_1 = self.df_offline[(self.df_offline.Discount_rate != 'null') & (self.df_offline.Date != 'null')][['Merchant_id', 'Coupon_id']]
        df_ft6_1 = df_ft6_1.drop_duplicates()
        df_ft6_1 = df_ft6_1.groupby(['Merchant_id'], as_index=False)['Coupon_id'].count()
        df_ft6 = pd.merge(df_merchant, df_ft6_1, how='left', on=['Merchant_id'])
        df_ft6.fillna(0, inplace=True)
        df_ft6.rename(columns={'Coupon_id':'cp_used_types_in_merchant'}, inplace=True)
        
        df_ft6.set_index(['Merchant_id'], inplace=True)     # Merge use
        
        # [feature 7]
        df_ft7_1 = self.df_offline[(self.df_offline.Coupon_id != 'null')][['Merchant_id', 'Coupon_id']]
        df_ft7_1 = df_ft7_1.drop_duplicates()
        df_ft7_1 = df_ft7_1.groupby(['Merchant_id'], as_index=False)['Coupon_id'].count()
        df_ft7_1.rename(columns={'Coupon_id':'cp_types_in_merchant'}, inplace=True)
        
        df_ft7 = pd.merge(df_ft6, df_ft7_1, how='left', on=['Merchant_id'])
        df_ft7.fillna(0, inplace=True)
        df_ft7['cp_used_types_outof_all_types_in_merchant'] = df_ft7.apply(lambda x : x[1]/x[2] if x[2] != 0 else 0, axis=1)
        df_ft7 = df_ft7[['Merchant_id', 'cp_used_types_outof_all_types_in_merchant']]
        
        df_ft7.set_index(['Merchant_id'], inplace=True)     # Merge use
        
        # [feature 5]
        df_ft5_1 = self.df_offline[(self.df_offline.Discount_rate != 'null') & (self.df_offline.Date != 'null')][['Merchant_id', 'User_id']]
        df_ft5_1 = df_ft5_1.drop_duplicates()
        df_ft5_1 = df_ft5_1.groupby(['Merchant_id'], as_index=False)['User_id'].count()
        
        df_ft5_2 = self.df_offline[(self.df_offline.Discount_rate != 'null') & (self.df_offline.Date != 'null')][['Merchant_id', 'Coupon_id']]
        df_ft5_2 = df_ft5_2.groupby(['Merchant_id'], as_index=False)['Coupon_id'].count()
        
        df_ft5_3 = pd.merge(pd.merge(df_merchant, df_ft5_1, how='left', on=['Merchant_id']), df_ft5_2, how='left', on=['Merchant_id'])
        df_ft5_3.fillna(0, inplace=True)
        
        df_ft5_3['cp_used_num_per_user_in_merchant'] = df_ft5_3.apply(lambda x : x[2]/x[1] if x[1] != 0 else 0, axis=1)
        df_ft5 = df_ft5_3[['Merchant_id', 'cp_used_num_per_user_in_merchant']]
        
        df_ft5.set_index(['Merchant_id'], inplace=True)     # Merge use
        
        # [feature 4]
        df_feature4_1 = self.df_offline[(self.df_offline.Discount_rate != 'null') & (self.df_offline.Date != 'null')][['Merchant_id', 'Discount_rate']]
        df_feature4_1.iloc[:,-1] = df_feature4_1.Discount_rate.apply(lambda x:x if ':' not in x 
                                                                     else (float(str(x).split(':')[0]) - float(str(x).split(':')[1]))/float(str(x).split(':')[0]))
        df_feature4_1.iloc[:,-1] = df_feature4_1.Discount_rate.astype('float')
        df_feature4_2 = df_feature4_1.groupby(['Merchant_id'], as_index=False)['Discount_rate'].mean()
        df_feature4_2.rename(columns={'Discount_rate':'cp_used_discount_mean_in_merchant'}, inplace=True)
        
        df_feature4_3 = df_feature4_1.groupby(['Merchant_id'], as_index=False)['Discount_rate'].max()
        df_feature4_3.rename(columns={'Discount_rate':'cp_used_discount_max_in_merchant'}, inplace=True)
        
        df_feature4_4 = df_feature4_1.groupby(['Merchant_id'], as_index=False)['Discount_rate'].min()
        df_feature4_4.rename(columns={'Discount_rate':'cp_used_discount_min_in_merchant'}, inplace=True)
        
        df_feature4 = pd.merge(df_feature4_2, df_feature4_3, on=['Merchant_id'])
        df_feature4 = pd.merge(df_feature4, df_feature4_4, on=['Merchant_id'])
        
        df_ft4 = pd.merge(df_merchant, df_feature4, how='left', on=['Merchant_id'])
        df_ft4.fillna(0, inplace=True)
        
        df_ft4.set_index(['Merchant_id'], inplace=True)     # Merge use
        
        # [feature 3]
        df_ft3_1 = self.df_offline[(self.df_offline.Coupon_id != 'null') & (self.df_offline.Date != 'null')][['Merchant_id', 'Coupon_id']]
        df_ft3_1 = df_ft3_1.groupby(['Merchant_id'], as_index=False)['Coupon_id'].count()
        df_ft3_1.rename(columns={'Coupon_id':'cp_used_num_in_merchant'}, inplace=True)
        df_ft3 = pd.merge(df_merchant, df_ft3_1, how='left', on=['Merchant_id'])
        df_ft3.fillna(0, inplace=True)
        df_ft3_copy = df_ft3
        df_ft3.set_index(['Merchant_id'], inplace=True)     # Merge use
        
        # [feature 8]
        df_ft8_1 = self.df_offline[(self.df_offline.Coupon_id != 'null')][['Merchant_id', 'Coupon_id']]
        df_ft8_1 = df_ft8_1.drop_duplicates()
        df_ft8_1 = df_ft8_1.groupby(['Merchant_id'], as_index=False)['Coupon_id'].count()
        
        df_ft8 = pd.merge(df_ft3, df_ft8_1, how='left', on=['Merchant_id'])
        df_ft8.fillna(0, inplace=True)
        df_ft8['cp_used_num_per_type_in_merchant'] = df_ft8.apply(lambda x : x[1]/x[2] if x[2] != 0 else 0, axis=1)
        df_ft8 = df_ft8[['Merchant_id', 'cp_used_num_per_type_in_merchant']]
        
        df_ft8.set_index(['Merchant_id'], inplace=True)     # Merge use
        
        # [feature 2]
        df_ft2_1 = self.df_offline[(self.df_offline.Coupon_id != 'null') & (self.df_offline.Date == 'null')][['Merchant_id', 'Coupon_id']]
        df_ft2_1 = df_ft2_1.groupby(['Merchant_id'], as_index=False)['Coupon_id'].count()
        df_ft2_1.rename(columns={'Coupon_id':'cp_not_used_num_in_merchant'}, inplace=True)
        df_ft2 = pd.merge(df_merchant, df_ft2_1, how='left', on=['Merchant_id'])
        df_ft2.fillna(0, inplace=True)
        
        df_ft2.set_index(['Merchant_id'], inplace=True)     # Merge use
        
        # [feature 1]
        df_ft1_1 = self.df_offline[(self.df_offline.Coupon_id != 'null')][['Merchant_id', 'Coupon_id']] 
        df_ft1_1 = df_ft1_1.groupby(['Merchant_id'], as_index=False)['Coupon_id'].count()
        df_ft1_1.rename(columns={'Coupon_id':'picked_cp_num_from_merchant'}, inplace=True)
        df_ft1 = pd.merge(df_merchant, df_ft1_1, how='left', on=['Merchant_id'])
        df_ft1.fillna(0, inplace=True)
        df_ft1_copy = df_ft1
        df_ft1.set_index(['Merchant_id'], inplace=True)     # Merge use
        
        # [feature 11]
        df_ft11 = pd.merge(df_ft1_copy, df_ft3_copy, how='inner', on=['Merchant_id'])
        df_ft11.reset_index(inplace=True)
        df_ft11['cp_used_rate_in_merchant'] = df_ft11.apply(lambda x : x[2]/x[1] if x[1] != 0 else 0, axis=1)
        df_ft11 = df_ft11[['Merchant_id', 'cp_used_rate_in_merchant']]
        
        df_ft11.set_index(['Merchant_id'], inplace=True)     # Merge use
        
        # Merge all single features
        df_ft = pd.concat([df_ft1, df_ft2, df_ft3, df_ft4, df_ft5, df_ft6, df_ft7, df_ft8, df_ft9, df_ft10, df_ft11], axis=1)
        df_ft.reset_index(inplace=True)
        df_ft.rename(columns={'index':'Merchant_id'}, inplace=True)
        df_ft.to_csv(os.path.join(self.feature_data_dir, 'merchant_features.csv'), index=False)
    
    def user_merchant_features(self):
        # user-merchant
        user_merchant = self.df_offline[['User_id', 'Merchant_id']]
        user_merchant.drop_duplicates(inplace=True)
        user_merchant.sort_values(by='User_id', inplace=True)
        
        
        # [feature 2]
        df_ft2_1 = self.df_offline[(self.df_offline.Coupon_id != 'null') & (self.df_offline.Date == 'null')][['User_id', 'Merchant_id', 'Coupon_id']] 
        df_ft2_1 = df_ft2_1.groupby(['User_id', 'Merchant_id'])['Coupon_id'].count()
        df_ft2_1 = pd.DataFrame(df_ft2_1)
        df_ft2_1.rename(columns={'Coupon_id':'user_cp_not_use_num_in_merchant'}, inplace=True)
        
        df_ft2 = pd.merge(user_merchant, df_ft2_1, how='left', on=['User_id', 'Merchant_id'])
        df_ft2.fillna(0, inplace=True)
        df_ft2_copy = df_ft2.copy()
        
        df_ft2.set_index(['User_id', 'Merchant_id'], inplace=True)     # Merge use
        
        # [feature 1]
        df_ft1_1 = self.df_offline[['User_id', 'Merchant_id', 'Coupon_id']] 
        df_ft1_1['Coupon_id'] = df_ft1_1.Coupon_id.apply(lambda x : None if x == 'null' else x)
        df_ft1 = df_ft1_1.groupby(['User_id', 'Merchant_id'])['Coupon_id'].count()
        df_ft1 = pd.DataFrame(df_ft1)
        df_ft1.rename(columns={'Coupon_id':'user_get_cp_num_in_merchant'}, inplace=True)
        df_ft1.reset_index(inplace=True)  
        df_ft1_copy = df_ft1
        ###### The feature1 has been covered ######
        
        # [feature 3]
        df_ft3_1 = self.df_offline[(self.df_offline.Coupon_id != 'null') & (self.df_offline.Date != 'null')][['User_id', 'Merchant_id', 'Coupon_id']] 
        df_ft3_1 = df_ft3_1.groupby(['User_id', 'Merchant_id'])['Coupon_id'].count()
        df_ft3_1 = pd.DataFrame(df_ft3_1)
        df_ft3_1.rename(columns={'Coupon_id':'user_cp_used_num_in_merchant'}, inplace=True)
        
        df_ft3 = pd.merge(user_merchant, df_ft3_1, how='left', on=['User_id', 'Merchant_id'])
        df_ft3.fillna(0, inplace=True)   
        
        df_ft3 = pd.merge(df_ft3, df_ft1_copy, on=['User_id', 'Merchant_id'])  
        df_ft3['user_cp_used_rate_in_merchant'] = df_ft3.apply(lambda x : x[2]/x[3] if x[3] != 0 else 0, axis=1)
        df_ft3_copy = df_ft3.copy()
        df_ft3_copy.reset_index(inplace=True)
        
        df_ft3.set_index(['User_id', 'Merchant_id'], inplace=True)     # Merge use
        
        # [feature 5,6]
        df_ft4_1 = self.df_offline[(self.df_offline.Coupon_id != 'null') & (self.df_offline.Date == 'null')][['User_id', 'Coupon_id']]
        df_ft4_1 = df_ft4_1.groupby(['User_id'])['Coupon_id'].count()
        df_ft4_1 = pd.DataFrame(df_ft4_1)
        df_ft4_1.reset_index(inplace=True)
        df_ft4_1.rename(columns={'Coupon_id':'user_total_cp_unused_num'}, inplace=True)
        
        df_ft4_2 = self.df_offline[(self.df_offline.Coupon_id != 'null') & (self.df_offline.Date != 'null')][['User_id', 'Coupon_id']]
        df_ft4_2 = df_ft4_2.groupby(['User_id'])['Coupon_id'].count()
        df_ft4_2 = pd.DataFrame(df_ft4_2)
        df_ft4_2.reset_index(inplace=True)
        df_ft4_2.rename(columns={'Coupon_id':'user_total_cp_used_num'}, inplace=True)

        df_ft4_3 = pd.merge(df_ft2_copy, df_ft4_1, how='left', on=['User_id'])
        df_ft4_3.fillna(0, inplace=True)
        df_ft4_3['user_merchant_cp_unused_num_outof_all_user_unused_num'] = df_ft4_3.apply(lambda x : x[-2]/x[-1] if x[-1] != 0 else 0, axis=1)
        df_ft4_3 = df_ft4_3[['User_id', 'Merchant_id', 'user_merchant_cp_unused_num_outof_all_user_unused_num']]
        df_ft4_4 = pd.merge(df_ft3_copy, df_ft4_2, how='left', on=['User_id'])
        df_ft4_4.fillna(0, inplace=True)
        df_ft4_4['user_merchant_cp_used_num_outof_all_user_used_num'] = df_ft4_4.apply(lambda x : x[-4]/x[-1] if x[-1] != 0 else 0, axis=1)
        df_ft4_4 = df_ft4_4[['User_id', 'Merchant_id', 'user_merchant_cp_used_num_outof_all_user_used_num']]
        
        df_ft4 = pd.merge(df_ft4_3, df_ft4_4, on=['User_id', 'Merchant_id'])
        
        df_ft4.set_index(['User_id', 'Merchant_id'], inplace=True)     # Merge use
        
        # [feature 7,8]
        df_merchant = self.df_offline[['Merchant_id']].drop_duplicates()
        df_ft5_1 = self.df_offline[(self.df_offline.Coupon_id != 'null') & (self.df_offline.Date == 'null')][['Merchant_id', 'Coupon_id']]
        df_ft5_1 = df_ft5_1.groupby(['Merchant_id'], as_index=False)['Coupon_id'].count()
        df_ft5_1.rename(columns={'Coupon_id':'cp_not_used_num_in_merchant'}, inplace=True)
        df_ft5_1 = pd.merge(df_merchant, df_ft5_1, how='left', on=['Merchant_id'])
        df_ft5_1.fillna(0, inplace=True)
        
        df_ft5_2 = self.df_offline[(self.df_offline.Coupon_id != 'null') & (self.df_offline.Date != 'null')][['Merchant_id', 'Coupon_id']]
        df_ft5_2 = df_ft5_2.groupby(['Merchant_id'], as_index=False)['Coupon_id'].count()
        df_ft5_2.rename(columns={'Coupon_id':'cp_used_num_in_merchant'}, inplace=True)
        df_ft5_2 = pd.merge(df_merchant, df_ft5_2, how='left', on=['Merchant_id'])
        df_ft5_2.fillna(0, inplace=True)
        
        df_ft5_3 = pd.merge(df_ft2_copy, df_ft5_1, how='left', on=['Merchant_id'])
        df_ft5_3['user_merchant_cp_unused_num_outof_merchant_unused_num'] = df_ft5_3.apply(lambda x : x[-2]/x[-1] if x[-1] != 0 else 0, axis=1)
        df_ft5_3 = df_ft5_3[['User_id', 'Merchant_id', 'user_merchant_cp_unused_num_outof_merchant_unused_num']]
        
        df_ft5_4 = pd.merge(df_ft3_copy, df_ft5_2, how='left', on=['Merchant_id'])
        df_ft5_4['user_merchant_cp_used_num_outof_merchant_unused_num'] = df_ft5_4.apply(lambda x : x[-4]/x[-1] if x[-1] != 0 else 0, axis=1)
        df_ft5_4 = df_ft5_4[['User_id', 'Merchant_id', 'user_merchant_cp_used_num_outof_merchant_unused_num']]
        
        df_ft5 = pd.merge(df_ft5_3, df_ft5_4, on=['User_id', 'Merchant_id'])
        
        df_ft5.set_index(['User_id', 'Merchant_id'], inplace=True)     # Merge use
        
        # Merge all single features
        df_ft = pd.concat([df_ft2, df_ft3, df_ft4, df_ft5], axis=1)
        df_ft.reset_index(inplace=True)
        df_ft.to_csv(os.path.join(self.feature_data_dir, 'user_merchant_features.csv'), index=False)
    
    def coupon_features(self):
        # All coupons
        df_ft1 = self.df_offline[(self.df_offline.Coupon_id != 'null')][['Coupon_id', 'Discount_rate']]
        df_ft1.drop_duplicates(inplace=True)
        
        # Coupon type
        df_ft2 = df_ft1.copy()
        df_ft2['Discount_rate'] = df_ft2.Discount_rate.apply(lambda x : '1' if ':' in x else '0')
        df_ft2.rename(columns={'Discount_rate':'cp_type'}, inplace=True)
        
        # Coupon discount
        df_ft3 = df_ft1.copy()
        df_ft3['Discount_rate'] = df_ft3.Discount_rate.apply(lambda x:x if ':' not in x 
                                        else (float(str(x).split(':')[0]) - float(str(x).split(':')[1]))/float(str(x).split(':')[0]))
        df_ft3.rename(columns={'Discount_rate':'cp_discount'}, inplace=True)
        
        # The minimum consumption for coupon:1
        df_ft4 = df_ft1.copy()
        df_ft4['Discount_rate'] = df_ft4.Discount_rate.apply(lambda x : '0' if ':' not in x else (str(x).split(':')[0]))
        df_ft4.rename(columns={'Discount_rate':'min_consume'}, inplace=True)
        
        # Coupon frequency
        df_ft5 = self.df_offline[(self.df_offline.Coupon_id != 'null')][['Coupon_id', 'Discount_rate']]
        df_ft5 = df_ft5.groupby(['Coupon_id'])['Discount_rate'].count()
        df_ft5 = pd.DataFrame(df_ft5)
        df_ft5.reset_index(inplace=True)
        df_ft5.rename(columns={'Discount_rate':'cp_freq'}, inplace=True)
        
        # Coupon totally used number
        df_ft6_1 = self.df_offline[(self.df_offline.Coupon_id != 'null') & (self.df_offline.Date != 'null')][['Coupon_id', 'Date']]
        df_ft6_1 = df_ft6_1.groupby(['Coupon_id'])['Date'].count()
        df_ft6_1 = pd.DataFrame(df_ft6_1)
        df_ft6_1.reset_index(inplace=True)
        
        df_ft6_2 = df_ft1.copy()
        df_ft6 = pd.merge(df_ft6_2, df_ft6_1, how='left', on=['Coupon_id'])
        df_ft6.fillna(0, inplace=True)
        df_ft6.rename(columns={'Date':'cp_total_used_num'}, inplace=True)
        df_ft6.drop(['Discount_rate'], inplace=True, axis=1)
        
        # Coupon totally used rate
        df_ft7_1 = df_ft5.copy()
        df_ft7_2 = df_ft6.copy()
        df_ft7 = pd.merge(df_ft7_1, df_ft7_2)
        df_ft7['cp_total_used_rate'] = df_ft7.apply(lambda x : x[2]/x[1] if x[1] != 0 else 0, axis=1)
        df_ft7 = df_ft7[['Coupon_id', 'cp_total_used_rate']]
        
        # Coupon average used duration in total
        df_ft8_1 = self.df_offline[(self.df_offline.Coupon_id != 'null') & (self.df_offline.Date != 'null')][['Coupon_id', 'Date_received', 'Date']]
        df_ft8_1['Date_received'] = pd.to_datetime(df_ft8_1.Date_received)
        df_ft8_1['Date'] = pd.to_datetime(df_ft8_1.Date)
        df_ft8_1['cp_used_duration_total_mean'] = df_ft8_1.apply(lambda x : float((x[2]-x[1]).days), axis=1)
        df_ft8_1 = df_ft8_1.groupby(['Coupon_id'])['cp_used_duration_total_mean'].mean()
        df_ft8_1 = pd.DataFrame(df_ft8_1)
        df_ft8_1.reset_index(inplace=True)
        
        df_ft8_2 = df_ft1.copy()
        df_ft8 = pd.merge(df_ft8_2, df_ft8_1, how='left', on=['Coupon_id'])
        df_ft8.fillna(0, inplace=True)
        df_ft8 = df_ft8[['Coupon_id', 'cp_used_duration_total_mean']]
        
        # Coupon been received number
        df_ft9_1 = self.df_offline[(self.df_offline.Date_received != 'null')][['Coupon_id', 'Date_received']]
        df_ft9 = df_ft9_1.groupby(['Coupon_id'])['Date_received'].count()
        df_ft9 = pd.DataFrame(df_ft9)
        df_ft9.rename(columns={'Date_received':'cp_total_received_num'}, inplace=True)
        df_ft9.reset_index(inplace=True)
        
        # Coupon day of week
        df_ft10_1 = self.df_offline[(self.df_offline.Date_received != 'null')][['Coupon_id', 'Date_received']]
        df_ft10_1.drop_duplicates(inplace=True)
        df_ft10_1['Date_received'] = pd.to_datetime(df_ft10_1.Date_received)
        df_ft10_1['Date_received'] = df_ft10_1.Date_received.apply(lambda x : x.dayofweek)
        df_ft10_1.drop_duplicates(inplace=True)
        df_ft10_1 = df_ft10_1.groupby(['Coupon_id'])['Date_received'].apply(list)
        df_ft10 = pd.DataFrame(df_ft10_1)
        df_ft10.reset_index(inplace=True)
        df_ft10.rename(columns={'Date_received':'cp_received_dayofweek'}, inplace=True)
        
        # Coupon day of month
        df_ft11_1 = self.df_offline[(self.df_offline.Date_received != 'null')][['Coupon_id', 'Date_received']]
        df_ft11_1.drop_duplicates(inplace=True)
        df_ft11_1['Date_received'] = pd.to_datetime(df_ft11_1.Date_received)
        df_ft11_1['Date_received'] = df_ft11_1.Date_received.apply(lambda x : x.days_in_month)
        df_ft11_1.drop_duplicates(inplace=True)
        df_ft11_1 = df_ft11_1.groupby(['Coupon_id'])['Date_received'].apply(list)
        df_ft11 = pd.DataFrame(df_ft11_1)
        df_ft11.reset_index(inplace=True)
        df_ft11.rename(columns={'Date_received':'cp_received_dayofmonth'}, inplace=True)
        
        # Merge all single features
        for i in range(2,12):
            ss = "df_ft%d.set_index(['Coupon_id'], inplace=True)" % i       # Merge use index
            eval(ss)
        
        df_ft = pd.concat([df_ft2, df_ft3, df_ft4, df_ft5, df_ft6, df_ft7, df_ft8, df_ft9, df_ft10, df_ft11], axis=1)
        df_ft.reset_index(inplace=True)
        df_ft.rename(columns={'index':'Coupon_id'}, inplace=True)
        df_ft.to_csv(os.path.join(self.feature_data_dir, 'coupon_features.csv'), index=False)

    def integrate_all_features(self):
        """
        Merge all the features under self.feature_data_dir to integrate
        the training data for modeling
        Order: user -> merchant -> coupon
        """
        # Base relationships
        df1 = self.df_offline[(self.df_offline.Coupon_id != 'null')]
        df1['target'] = df1.apply(lambda x : '1' if x[-1] != 'null' else '0', axis=1)     # 1: used  0: not used
        df1.sort_values(by=['User_id'], inplace=True)
        df1.drop(['Date'], axis=1, inplace=True)
        df1['Distance'] = df1.Distance.apply(lambda x : '10' if x == 'null' else x)
        df1['Discount_rate'] = df1.Discount_rate.apply(lambda x:x if ':' not in x
                                else (float(str(x).split(':')[0]) - float(str(x).split(':')[1]))/float(str(x).split(':')[0]))
        
        # Offline user features/online user features
        offline_user = pd.read_csv(os.path.join(self.feature_data_dir, 'user_offline_features.csv'),
                                  dtype=str, 
                                  keep_default_na=False)
        
        online_user = pd.read_csv(os.path.join(self.feature_data_dir, 'user_online_features.csv'),
                                  dtype=str, 
                                  keep_default_na=False)
        user_features = pd.merge(offline_user, online_user, how='left', on=['User_id'])
        user_features.fillna(0, inplace=True)
        
        # Concatenate the user features
        df2 = pd.merge(df1, user_features, how='left', on=['User_id'])
        
        # Concatenate user-merchant features
        user_merchant = pd.read_csv(os.path.join(self.feature_data_dir, 'user_merchant_features.csv'),
                                  dtype=str, 
                                  keep_default_na=False)
        df3 = pd.merge(df2, user_merchant, how='left', on=['User_id', 'Merchant_id'])
        
        # Concatenate merchant features
        merchant = pd.read_csv(os.path.join(self.feature_data_dir, 'merchant_features.csv'),
                                  dtype=str, 
                                  keep_default_na=False)
        df4 = pd.merge(df3, merchant, how='left', on=['Merchant_id'])
        
        # Concatenate coupon features
        coupon = pd.read_csv(os.path.join(self.feature_data_dir, 'coupon_features.csv'),
                                  dtype=str, 
                                  keep_default_na=False)
        df5 = pd.merge(df4, coupon, how='left', on=['Coupon_id'])
        
        self.check_null(df5)
        
        df5.to_csv(os.path.join(self.feature_data_dir, 'o2o_train.csv'), index=False)        # Save the final training data
        
    def check_null(self, df):
        if df.isnull().values.any() is True or df.isna().values.any() is True:
            raise RuntimeError('There is null value in DataFrame')
    



if __name__ == '__main__':
    extr = extract('C:\\scnguh\\datamining\\o2o')
    
    extr.user_offline_features()
         
    extr.user_online_features()
          
    extr.merchant_features()
        
    extr.user_merchant_features()
      
    extr.coupon_features()
    
    extr.integrate_all_features()
    
    
    
    
    
    
