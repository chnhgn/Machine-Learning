# -*- coding: utf-8 -*-
'''
Created on Oct 15, 2018

@author: Eddy Hu
'''
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 300)
import datetime
import numpy as np
from sklearn.decomposition import PCA
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import timedelta


class extract(object):
    
    def __init__(self, file_dir, extr_type):
        self.df_train = pd.read_csv(file_dir + 'train_users_clean.csv')
        
        if extr_type == 'train':
            self.df_raw = self.df_train
        elif extr_type == 'test':
            self.df_raw = pd.read_csv(file_dir + 'test_users_clean.csv')
            
        self.df_age_gender_bkts = self.preprocess_age_gender_bkts(pd.read_csv(file_dir + 'age_gender_bkts.csv'))
        self.df_countries = pd.read_csv(file_dir + 'countries.csv')
        self.df_sessions = pd.read_csv(file_dir + 'sessions_clean.csv')
        self.extr_type = extr_type
    
    ''' 用户特征  '''
    
    # 用户年龄段
    def age_bucket(self):
        df = self.df_raw[['id', 'age', 'country_destination']]
        df['age_bucket'] = df.age.apply(lambda x:self.gen_age_bucket(x))
        return df[['id', 'age_bucket', 'country_destination']]
    
    # 首次操作与注册日期相差天数
    def first_active_account_created_delta(self):
        df = self.df_raw[['id', 'timestamp_first_active', 'date_account_created']]
        df.timestamp_first_active = df.timestamp_first_active.astype('str')
        df.timestamp_first_active = df.timestamp_first_active.apply(lambda x:datetime.datetime.strptime(x, '%Y%m%d%H%M%S').date())
        df.timestamp_first_active = pd.to_datetime(df.timestamp_first_active)
        df.date_account_created = pd.to_datetime(df.date_account_created)
        df['delta'] = df.apply(lambda x: x[2] - x[1], axis=1)
        df['first_active_account_created_delta'] = df.delta.apply(lambda x:float(x.days))
        return df[['id', 'first_active_account_created_delta']]
    
    # 页面操作次数
    def page_action_num(self):
        df1 = self.df_raw[['id']].drop_duplicates()
        df2 = self.df_sessions[['user_id', 'device_type']]
        df3 = pd.merge(df1, df2, left_on='id', right_on='user_id', how='left')
        df4 = df3.groupby(['id'], as_index=False)['device_type'].count()
        df4.rename(columns={'device_type':'page_action_num'}, inplace=True)
        return df4
    
    # 页面操作种类
    def page_action_type_num(self):
        df1 = self.df_raw[['id']].drop_duplicates()
        df2 = self.df_sessions[['user_id', 'action_type']]
        df3 = pd.merge(df1, df2, left_on='id', right_on='user_id', how='left')
        df3 = df3[['id', 'action_type']].drop_duplicates()
        df4 = df3.groupby(['id'], as_index=False)['action_type'].count()
        df4.rename(columns={'action_type':'page_action_type_num'}, inplace=True)
        return df4
    
    # 使用设备的种类
    def device_type_num(self):
        df1 = self.df_raw[['id']].drop_duplicates()
        df2 = self.df_sessions[['user_id', 'device_type']].drop_duplicates()
        df3 = pd.merge(df1, df2, left_on='id', right_on='user_id', how='left')
        df4 = df3.groupby(['id'], as_index=False)['device_type'].count()
        df4.rename(columns={'device_type':'device_type_num'}, inplace=True)
        return df4
    
    # 使用率最高的设备
    def frequent_used_device(self):
        df1 = self.df_raw[['id', 'first_device_type']]
        df1.rename(columns={'first_device_type':'frequent_used_device'}, inplace=True)
        return df1
    
    # 用户操作总消耗时间
    def user_time_consume(self):
        df1 = self.df_raw[['id']]
        df2 = self.df_sessions[['user_id', 'secs_elapsed']]
        df3 = pd.merge(df1, df2, left_on='id', right_on='user_id', how='left')
        df4 = df3.groupby(['id'], as_index=False).agg({'secs_elapsed':'sum'})
        df4.rename(columns={'secs_elapsed':'user_time_consume'}, inplace=True)
        return df4
    
    # 用户平均每种操作类型消耗的时间
    def user_time_avg(self):
        df1 = self.df_raw[['id']]
        df2 = self.df_sessions[['user_id', 'action', 'secs_elapsed']]
        df3 = pd.merge(df1, df2, left_on='id', right_on='user_id', how='left')
        df4 = df3.groupby(['id'], as_index=False).agg({'secs_elapsed':'sum'})
        df5 = df3.groupby(['id'], as_index=False).agg({'action':'count'})
        df6 = pd.merge(df4, df5, on='id')
        df6['user_time_avg'] = df6.apply(lambda x:0 if x[2] == 0 else x[1] / x[2], axis=1)
        return df6[['id', 'user_time_avg']]
    
    # 用户最频繁的操作
    def frequent_action(self):
        df1 = self.df_raw[['id']]
        df2 = self.df_sessions[['user_id', 'action', 'device_type']]
        df3 = pd.merge(df1, df2, left_on='id', right_on='user_id', how='left')
        df4 = df3.groupby(['id', 'action'], as_index=False).agg({'device_type':'count'})
        df4.sort_values(['id', 'device_type'], ascending=[1, 0], inplace=True)
        df5 = df4.groupby(['id'], as_index=False).head(1)
        df5.rename(columns={'action':'frequent_action'}, inplace=True)
        df5 = pd.merge(df1, df5, on='id', how='left')
        df5.fillna('unknown', inplace=True)
        return df5[['id', 'frequent_action']]
    
    # 是否发起过booking_request或者booking_response操作
    def has_booking_action(self):
        df1 = self.df_raw[['id']]
        df2 = self.df_sessions[['user_id', 'action_type']]
        df3 = pd.merge(df1, df2, left_on='id', right_on='user_id')
        df4 = df3[['id', 'action_type']].drop_duplicates()
        df4 = df4[(df4.action_type == 'booking_request') | (df4.action_type == 'booking_response')][['id', 'action_type']]
        df4['has_booking_action'] = 1
        df5 = pd.merge(df3, df4, on='id', how='left')
        df5 = df5[['id', 'has_booking_action']].drop_duplicates()
        df5.fillna(0, inplace=True)
        df6 = pd.merge(df1, df5, on='id', how='left')
        df6.fillna(0, inplace=True)
        return df6
    
    # 年龄是否小于26岁
    def is_student(self):
        df1 = self.df_raw[['id', 'age']]
        df1.age = df1.age.astype('int')
        df1['is_student'] = df1.apply(lambda x:'1' if x['age'] <= 26 else '0', axis=1)
        return df1[['id', 'is_student']]
    
    ''' 年龄层特征 '''
    
    # 各层预订率
    def age_bucket_booking_rate(self):
        df1 = self.df_train[['age', 'country_destination']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df2 = df1.groupby(['age'], as_index=False).agg({'country_destination':'count'})
        df2.rename(columns={'country_destination':'total'}, inplace=True)
        df3 = df1[(df1.country_destination != 'NDF')]
        df4 = df3.groupby(['age'], as_index=False).agg({'country_destination':'count'})
        df4.rename(columns={'country_destination':'booking'}, inplace=True)
        df5 = pd.merge(df2, df4, on='age')
        df5['age_bucket_booking_rate'] = df5.apply(lambda x:x[2] / x[1], axis=1)
        
        return df5[['age', 'age_bucket_booking_rate']]
    
    # 各层预定最多的目的地
    def most_popular_place(self):
        df1 = self.df_train[['age', 'country_destination', 'date_account_created']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df1 = df1[(df1.country_destination != 'NDF')]
        df2 = df1.groupby(['age', 'country_destination'], as_index=False).agg({'date_account_created':'count'})
        df2.rename(columns={'date_account_created':'total'}, inplace=True)
        df2.sort_values(['age', 'total'], ascending=[1, 0], inplace=True)
        df3 = df2.groupby(['age'], as_index=False).head(1)
        df3.rename(columns={'country_destination':'most_popular_place'}, inplace=True)
        return df3[['age', 'most_popular_place']]
    
    # 各层预定最少的目的地
    def least_popular_place(self):
        df1 = self.df_train[['age', 'country_destination', 'date_account_created']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df2 = df1.groupby(['age', 'country_destination'], as_index=False).agg({'date_account_created':'count'})
        df2.rename(columns={'date_account_created':'total'}, inplace=True)
        df2.sort_values(['age', 'total'], ascending=[1, 1], inplace=True)
        df3 = df2.groupby(['age'], as_index=False).head(1)
        df3.rename(columns={'country_destination':'least_popular_place'}, inplace=True)
        return df3[['age', 'least_popular_place']]
    
    # 各层各个目的地的预定比重
    def bucket_place_booking_rate(self):
        df1 = self.df_train[['age', 'country_destination', 'date_account_created']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df2 = df1.groupby(['age', 'country_destination'], as_index=False).agg({'date_account_created':'count'})
        df2.rename(columns={'date_account_created':'subtotal'}, inplace=True)
        
        df3 = df1.groupby(['age'], as_index=False).agg({'country_destination':'count'})
        df3.rename(columns={'country_destination':'total'}, inplace=True)
        df4 = pd.merge(df3, df2, on='age', how='left')
        df4['rate'] = df4.apply(lambda x:x[3] / x[1], axis=1)
        df4.drop(['total', 'subtotal'], inplace=True, axis=1)
        df5 = df4.pivot(index='age', columns='country_destination', values='rate')
        df5.fillna(0, inplace=True)
        df5.reset_index(inplace=True)
        df5.drop(['NDF'], axis=1, inplace=True)
        for index, value in df5.dtypes.iteritems():
            if index is not 'age':
                df5.rename(columns={index:index + '_booking_rate'}, inplace=True)
        return df5
    
    # 各层预定最多目的地的次数
    def bucket_booking_place_max_num(self):
        df1 = self.df_train[['age', 'country_destination', 'date_account_created']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df2 = df1.groupby(['age', 'country_destination'], as_index=False).agg({'date_account_created':'count'})
        df2.rename(columns={'date_account_created':'subtotal'}, inplace=True)
        df2 = df2[df2.country_destination != 'NDF']
        df2.sort_values(['age', 'subtotal'], ascending=[1, 0], inplace=True)
        df2 = df2.groupby(['age'], as_index=False).head(1)[['age', 'subtotal']]
        df2.rename(columns={'subtotal':'bucket_booking_place_max_num'}, inplace=True)
        return df2   
    
    # 各层预定最少目的地的次数
    def bucket_booking_place_min_num(self):
        df1 = self.df_train[['age', 'country_destination', 'date_account_created']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df2 = df1.groupby(['age', 'country_destination'], as_index=False).agg({'date_account_created':'count'})
        df2.rename(columns={'date_account_created':'subtotal'}, inplace=True)
        df2 = df2[df2.country_destination != 'NDF']
        df2.sort_values(['age', 'subtotal'], ascending=[1, 1], inplace=True)
        df2 = df2.groupby(['age'], as_index=False).head(1)[['age', 'subtotal']]
        df2.rename(columns={'subtotal':'bucket_booking_place_min_num'}, inplace=True)
        return df2
    
    # 各层不预定的比重
    def bucket_no_booking_rate(self):
        df1 = self.df_train[['age', 'country_destination']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df2 = df1.groupby(['age'], as_index=False).agg({'country_destination':'count'})
        df2.rename(columns={'country_destination':'total'}, inplace=True)
        
        df3 = df1[df1.country_destination == 'NDF']
        df3 = df3.groupby(['age'], as_index=False).agg({'country_destination':'count'})
        df3.rename(columns={'country_destination':'subtotal'}, inplace=True)
        
        df4 = pd.merge(df2, df3, on='age')
        df4['bucket_no_booking_rate'] = df4.apply(lambda x:x[2] / x[1], axis=1)
        return df4[['age', 'bucket_no_booking_rate']]
    
    # 各层性别的比重
    def bucket_gender_rate(self):
        df1 = self.df_raw[['age', 'gender', 'date_account_created']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df2 = df1.groupby(['age', 'gender'], as_index=False).agg({'date_account_created':'count'})
        
        df3 = df1.groupby(['age'], as_index=False).agg({'date_account_created':'count'})
        df4 = pd.merge(df2, df3, on='age', how='left')
        df4['bucket_gender_rate'] = df4.apply(lambda x:x[2] / x[3], axis=1)
        df4 = df4[['age', 'gender', 'bucket_gender_rate']]
        df5 = df4.pivot(index='age', columns='gender', values='bucket_gender_rate')
        df5.reset_index(inplace=True)
        
        for index, value in df5.dtypes.iteritems():
            if index is not 'age':
                df5.rename(columns={index:index + '_gender_rate'}, inplace=True)
                
        return df5
    
    # 各层各个设备的使用率
    def bucket_device_rate(self):
        df1 = self.df_raw[['age', 'first_device_type', 'date_account_created']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df2 = df1.groupby(['age', 'first_device_type'], as_index=False).agg({'date_account_created':'count'})
        
        df3 = df1.groupby(['age'], as_index=False).agg({'date_account_created':'count'})
        df4 = pd.merge(df2, df3, on='age', how='left')
        df4['bucket_device_rate'] = df4.apply(lambda x:x[2] / x[3], axis=1)
        df5 = df4.pivot('age', 'first_device_type', 'bucket_device_rate')
        df5.fillna(0, inplace=True)
        df5.reset_index(inplace=True)
        for index, value in df5.dtypes.iteritems():
            if index is not 'age':
                df5.rename(columns={index:index + '_device_rate'}, inplace=True)
        
        return df5
    
    # 各层各个浏览器的使用率
    def bucket_browser_rate(self):
        df1 = self.df_raw[['age', 'first_browser', 'date_account_created']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df2 = df1.groupby(['age', 'first_browser'], as_index=False).agg({'date_account_created':'count'})
        
        df3 = df1.groupby(['age'], as_index=False).agg({'date_account_created':'count'})
        df4 = pd.merge(df2, df3, on='age', how='left')
        df4['bucket_browser_rate'] = df4.apply(lambda x:x[2] / x[3], axis=1)
        df5 = df4.pivot('age', 'first_browser', 'bucket_browser_rate')
        df5.fillna(0, inplace=True)
        df5.reset_index(inplace=True)
        for index, value in df5.dtypes.iteritems():
            if index is not 'age':
                df5.rename(columns={index:index + '_browser_rate'}, inplace=True)
        
        return df5
    
    # 各层各个注册应用的使用率
    def bucket_signup_app_rate(self):
        df1 = self.df_raw[['age', 'signup_app', 'date_account_created']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df2 = df1.groupby(['age', 'signup_app'], as_index=False).agg({'date_account_created':'count'})
        
        df3 = df1.groupby(['age'], as_index=False).agg({'date_account_created':'count'})
        df4 = pd.merge(df2, df3, on='age', how='left')
        df4['bucket_signup_app_rate'] = df4.apply(lambda x:x[2] / x[3], axis=1)
        df5 = df4.pivot('age', 'signup_app', 'bucket_signup_app_rate')
        df5.fillna(0, inplace=True)
        df5.reset_index(inplace=True)
        for index, value in df5.dtypes.iteritems():
            if index is not 'age':
                df5.rename(columns={index:index + '_signup_app_rate'}, inplace=True)
        
        return df5
    
    # 各层各个注册方式的比重
    def bucket_signup_method_rate(self):
        df1 = self.df_raw[['age', 'signup_method', 'date_account_created']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df2 = df1.groupby(['age', 'signup_method'], as_index=False).agg({'date_account_created':'count'})
        
        df3 = df1.groupby(['age'], as_index=False).agg({'date_account_created':'count'})
        df4 = pd.merge(df2, df3, on='age', how='left')
        df4['bucket_signup_method_rate'] = df4.apply(lambda x:x[2] / x[3], axis=1)
        df5 = df4.pivot('age', 'signup_method', 'bucket_signup_method_rate')
        df5.fillna(0, inplace=True)
        df5.reset_index(inplace=True)
        for index, value in df5.dtypes.iteritems():
            if index is not 'age':
                df5.rename(columns={index:index + '_signup_method_rate'}, inplace=True)
        
        return df5
    
    # 各层信息渠道类型比重
    def bucket_affiliate_rate(self):
        df1 = self.df_raw[['age', 'first_affiliate_tracked', 'date_account_created']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df2 = df1.groupby(['age', 'first_affiliate_tracked'], as_index=False).agg({'date_account_created':'count'})
        
        df3 = df1.groupby(['age'], as_index=False).agg({'date_account_created':'count'})
        df4 = pd.merge(df2, df3, on='age', how='left')
        df4['bucket_affiliate_rate'] = df4.apply(lambda x:x[2] / x[3], axis=1)
        df5 = df4.pivot('age', 'first_affiliate_tracked', 'bucket_affiliate_rate')
        df5.fillna(0, inplace=True)
        df5.reset_index(inplace=True)
        for index, value in df5.dtypes.iteritems():
            if index is not 'age':
                df5.rename(columns={index:index + '_affiliate_rate'}, inplace=True)
        
        return df5
    
    # 各层信息渠道提供商比重
    def bucket_affiliate_provider_rate(self):
        df1 = self.df_raw[['age', 'affiliate_provider', 'date_account_created']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df2 = df1.groupby(['age', 'affiliate_provider'], as_index=False).agg({'date_account_created':'count'})
        
        df3 = df1.groupby(['age'], as_index=False).agg({'date_account_created':'count'})
        df4 = pd.merge(df2, df3, on='age', how='left')
        df4['bucket_affiliate_provider_rate'] = df4.apply(lambda x:x[2] / x[3], axis=1)
        df5 = df4.pivot('age', 'affiliate_provider', 'bucket_affiliate_provider_rate')
        df5.fillna(0, inplace=True)
        df5.reset_index(inplace=True)
        for index, value in df5.dtypes.iteritems():
            if index is not 'age':
                df5.rename(columns={index:index + '_affiliate_provider_rate'}, inplace=True)
        
        return df5
        
    # 各层不同国家的性别比例
    def bucket_dest_gender_rate(self):
        df1 = self.df_age_gender_bkts
        df2 = df1.groupby(['age_bucket', 'country_destination', 'gender'], as_index=False).agg({'population_in_thousands':'sum'})
        df3 = df2.groupby(['age_bucket'])
        
        df = pd.DataFrame(columns=['age', 'country_destination', 'bucket_dest_gender_rate'])
        for name, group in df3:
            tmp = group.pivot('country_destination', 'gender', 'population_in_thousands')
            tmp.reset_index(inplace=True)
            tmp['bucket_dest_gender_rate'] = tmp.apply(lambda x:x['female'] / x['male'] if x['male'] != 0 else x['female'], axis=1)
            tmp['age'] = name
            tmp = tmp[['age', 'country_destination', 'bucket_dest_gender_rate']]
            df = df.append(tmp)
        
        df4 = df.pivot('age', 'country_destination', 'bucket_dest_gender_rate')
        df4.reset_index(inplace=True)
        for index, value in df4.dtypes.iteritems():
            if index is not 'age':
                df4.rename(columns={index:index + '_dest_gender_rate'}, inplace=True)
        
        return df4
    
    # 各层不同国家的人数
    def bucket_dest_population(self):
        df1 = self.df_age_gender_bkts
        df2 = df1.groupby(['age_bucket', 'country_destination'], as_index=False).agg({'population_in_thousands':'sum'})
        df3 = df2.pivot('age_bucket', 'country_destination', 'population_in_thousands')
        df3.reset_index(inplace=True)
        df3.rename(columns={'age_bucket':'age'}, inplace=True)
        for index, value in df3.dtypes.iteritems():
            if index is not 'age':
                df3.rename(columns={index:index + '_dest_population'}, inplace=True)
        
        return df3
        
    # 各层语言占比
    def bucket_lang_rate(self):
        df1 = self.df_raw[['age', 'language', 'date_account_created']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df2 = df1.groupby(['age', 'language'], as_index=False).agg({'date_account_created':'count'})
        df3 = df1.groupby(['age'], as_index=False).agg({'date_account_created':'count'})
        df4 = pd.merge(df2, df3, on='age', how='left')
        df4['bucket_lang_rate'] = df4.apply(lambda x:x[2] / x[3], axis=1)
        df4 = df4[['age', 'language', 'bucket_lang_rate']]
        df5 = df4.pivot('age', 'language', 'bucket_lang_rate')
        df5.reset_index(inplace=True)
        df5.fillna(0, inplace=True)
        for index, value in df5.dtypes.iteritems():
            if index is not 'age':
                df5.rename(columns={index:index + '_lang_rate'}, inplace=True)
        
        return df5
    
    # 各层目的地平均距离
    def bucket_dest_avg_dist(self):
        df1 = self.df_train[['age', 'country_destination']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df2 = self.df_countries
        df3 = pd.merge(df1, df2, on='country_destination', how='left')
        df3 = df3[['age', 'country_destination', 'distance_km']]
        df4 = df3.groupby(['age'], as_index=False).agg({'distance_km':'mean'})
        df4.rename(columns={'distance_km':'bucket_dest_avg_dist'}, inplace=True)
        return df4
    
    # 各层各个国家平均预定时长
    def bucket_booking_avg_time(self):
        df1 = self.df_train[['age', 'country_destination', 'date_account_created', 'date_first_booking']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        df1.date_account_created = pd.to_datetime(df1.date_account_created)
        df1.date_first_booking = pd.to_datetime(df1.date_first_booking)
        df1['bucket_booking_avg_time'] = df1.apply(lambda x:x[3] - x[2], axis=1)
        df2 = df1[['age', 'country_destination', 'bucket_booking_avg_time']]
        df2.bucket_booking_avg_time = df2.bucket_booking_avg_time.apply(lambda x:abs(x.days))
        df3 = df2.groupby(['age', 'country_destination'], as_index=False).agg({'bucket_booking_avg_time':'mean'})
        df3 = df3.pivot('age', 'country_destination', 'bucket_booking_avg_time')
        df3.fillna(999999, inplace=True)
        df3.reset_index(inplace=True)
        df3.drop(['NDF'], axis=1, inplace=True)
        for index, value in df3.dtypes.iteritems():
            if index is not 'age':
                df3.rename(columns={index:index + '_booking_avg_day'}, inplace=True)
        return df3
    
    # 各层当天预订率
    def bucket_intraday_booking_rate(self):
        df1 = self.df_train[['age', 'date_account_created', 'date_first_booking']]
        df1.age = df1.age.apply(lambda x:self.gen_age_bucket(x))
        
        df2 = df1[df1.date_account_created == df1.date_first_booking]
        df3 = df1.groupby(['age'], as_index=False).agg({'date_first_booking':'count'})
        df4 = df2.groupby(['age'], as_index=False).agg({'date_account_created':'count'})
        df5 = pd.merge(df3, df4, on='age', how='left')
        df5['bucket_intraday_booking_rate'] = df5.apply(lambda x:x[2] / x[1], axis=1)
        df5 = df5[['age', 'bucket_intraday_booking_rate']]
        df5.fillna(0, inplace=True)
        return df5
    
    # 注册日期是否节假日
    def near_holiday(self):
        calendar = USFederalHolidayCalendar()
        holidays = calendar.holidays()
        df1 = self.df_raw[['id', 'date_account_created']]
        df1.date_account_created = pd.to_datetime(df1.date_account_created)
        df1.date_account_created = pd.to_datetime(df1.date_account_created.dt.date)
        df1['near_holiday'] = (df1.date_account_created.isin(holidays + timedelta(days=1)) | df1.date_account_created.isin(holidays - timedelta(days=1)))
        return df1[['id', 'near_holiday']]
    
    # 注册日期是否为周末
    def is_weekend(self):
        df1 = self.df_raw[['id', 'date_account_created']]     
        df1.date_account_created = pd.to_datetime(df1.date_account_created)
        df1.date_account_created = df1.date_account_created.apply(lambda x:x.dayofweek)
        df1['is_weekend'] = df1.apply(lambda x : 1 if x['date_account_created'] in [5, 6] else 0, axis=1)
        return df1[['id', 'is_weekend']]
    
    # 年龄小于26岁的预订率
    def stu_booking_rate(self):
        df11 = self.df_train[['id', 'age']]
        df11.age = df11.age.astype('int')
        df11['is_student'] = df11.apply(lambda x:'1' if x['age'] <= 26 else '0', axis=1)
        df1 = df11[['id', 'is_student']]
        
        df2 = self.df_train[['id', 'country_destination']] 
        df3 = pd.merge(df1, df2, on='id')
        df4 = df3.groupby(['is_student'], as_index=False).agg({'country_destination':'count'})
        df5 = df3[df3.country_destination != 'NDF']
        df5 = df5.groupby(['is_student'], as_index=False).agg({'country_destination':'count'})
        df6 = pd.merge(df4, df5, on='is_student', how='left')
        df6['stu_booking_rate'] = df6.apply(lambda x : x[2] / x[1], axis=1)
        return df6[['is_student', 'stu_booking_rate']]
    
    # 年龄小于26岁选择各个目的地的比重
    def stu_dest_booking_rate(self):
        df11 = self.df_train[['id', 'age']]
        df11.age = df11.age.astype('int')
        df11['is_student'] = df11.apply(lambda x:'1' if x['age'] <= 26 else '0', axis=1)
        df1 = df11[['id', 'is_student']]
        
        df2 = self.df_train[['id', 'country_destination', 'date_account_created']]
        df3 = pd.merge(df1, df2, on='id')
        df4 = df3.groupby(['is_student'], as_index=False).agg({'country_destination':'count'})
        df5 = df3.groupby(['is_student', 'country_destination'], as_index=False).agg({'date_account_created':'count'})
        df6 = pd.merge(df5, df4, on='is_student', how='left')
        df6['stu_dest_booking_rate'] = df6.apply(lambda x : x[2] / x[3], axis=1)
        df6 = df6.pivot('is_student', 'country_destination_x', 'stu_dest_booking_rate')
        df6.reset_index(inplace=True)
        df6.drop(['NDF'], axis=1, inplace=True)
        for index, value in df6.dtypes.iteritems():
            if index is not 'is_student':
                df6.rename(columns={index:index + '_stu_dest_booking_rate'}, inplace=True)
        
        return df6
    
    # 各个季节各个目的地的预定率
    def season_dest_booking_rate(self):
        df1 = self.df_train[['date_account_created', 'country_destination']]
        df1.date_account_created = pd.to_datetime(df1.date_account_created)
        df1['month'] = df1.date_account_created.dt.month
        df1['season'] = df1.apply(lambda x : self.gen_season(x['month']), axis=1)
        df2 = df1.groupby(['season', 'country_destination'], as_index=False).agg({'date_account_created':'count'})
        df3 = df1.groupby(['season'], as_index=False).agg({'date_account_created':'count'})
        df4 = pd.merge(df2, df3, on='season', how='left')
        df4['season_dest_booking_rate'] = df4.apply(lambda x : x[2] / x[3], axis=1)
        df5 = df4.pivot('season', 'country_destination', 'season_dest_booking_rate')
        df5.drop(['NDF'], axis=1, inplace=True)
        df5.reset_index(inplace=True)
        for index, value in df5.dtypes.iteritems():
            if index is not 'season':
                df5.rename(columns={index:index + '_season_dest_booking_rate'}, inplace=True)
        
        return df5
        
    
    # 不同月份各个目的地的预订率
    def month_dest_booking_rate(self):
        df1 = self.df_train[['date_account_created', 'country_destination']]
        df1.date_account_created = pd.to_datetime(df1.date_account_created)
        df1['month'] = df1.date_account_created.dt.month
        df2 = df1.groupby(['month'], as_index=False).agg({'country_destination':'count'})
        df3 = df1.groupby(['month', 'country_destination'], as_index=False).agg({'date_account_created':'count'})
        df4 = pd.merge(df3, df2, on='month', how='left')
        df4['month_dest_booking_rate'] = df4.apply(lambda x : x[2] / x[3], axis=1)
        df5 = df4.pivot('month', 'country_destination_x', 'month_dest_booking_rate')
        df5.drop(['NDF'], axis=1, inplace=True)
        df5.reset_index(inplace=True)
        for index, value in df5.dtypes.iteritems():
            if index is not 'month':
                df5.rename(columns={index:index + '_month_dest_booking_rate'}, inplace=True)
        
        return df5
    
    
    ''' 特征整合 '''
        
    def features_user(self):
        df1 = self.age_bucket()
        df2 = self.first_active_account_created_delta()
        df3 = self.page_action_num()
        df4 = self.page_action_type_num()
        df5 = self.device_type_num()
        df6 = self.frequent_used_device()
        df7 = self.user_time_consume()
        df8 = self.user_time_avg()
#         df9 = self.frequent_action()
        df10 = self.has_booking_action()
        df11 = self.is_student()
        
        df1.set_index(['id'], inplace=True)
        df2.set_index(['id'], inplace=True)
        df3.set_index(['id'], inplace=True)
        df4.set_index(['id'], inplace=True)
        df5.set_index(['id'], inplace=True)
        df6.set_index(['id'], inplace=True)
        df7.set_index(['id'], inplace=True)
        df8.set_index(['id'], inplace=True)
#         df9.set_index(['id'], inplace=True)
        df10.set_index(['id'], inplace=True)
        df11.set_index(['id'], inplace=True)
        
        df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df10, df11], axis=1)
        df.reset_index(inplace=True)
        df.rename(columns={'index':'id'}, inplace=True)
        
        # 数值型归一化
        dfFinal = self.normalize(df)
        return dfFinal
    
    def features_age_bucket(self):

        df1 = self.bucket_gender_rate()
        
        # 网络生活取向
        cp1 = self.bucket_device_rate()
        cp2 = self.bucket_browser_rate()
        cp3 = self.bucket_signup_app_rate()
        cp4 = self.bucket_signup_method_rate()
          
        cp11 = pd.merge(cp1, cp2, on='age')
        cp22 = pd.merge(cp3, cp4, on='age')
        cp = pd.merge(cp11, cp22, on='age')
        cp.reset_index(inplace=True)
        cp = cp.iloc[:, 1:]
        cp.drop(['age'], inplace=True, axis=1)
        pca = PCA(n_components=5)
        cp_pca = pca.fit_transform(cp)
        df2 = pd.DataFrame(cp_pca, columns=['web_pca_1', 'web_pca_2', 'web_pca_3', 'web_pca_4', 'web_pca_5'])
        df2.loc[:, 'age'] = cp1.age
        
        # 信息渠道
        cp1 = self.bucket_affiliate_rate()
        cp2 = self.bucket_affiliate_provider_rate()
        cp = pd.merge(cp1, cp2, on='age')
        cp.reset_index(inplace=True)
        cp = cp.iloc[:, 1:]
        cp.drop(['age'], axis=1, inplace=True)
        pca = PCA(n_components=5)
        cp_pca = pca.fit_transform(cp)
        df3 = pd.DataFrame(cp_pca, columns=['aff_pca_1', 'aff_pca_2', 'aff_pca_3', 'aff_pca_4', 'aff_pca_5'])
        df3.loc[:, 'age'] = cp1.age
        
        df4 = self.bucket_lang_rate()
        
        df = pd.concat([df1, df2, df3, df4], axis=1)
        df.reset_index(inplace=True)
        key_col = df1.age
        df.drop(['age', 'index'], axis=1, inplace=True)
        df = pd.concat([df, key_col], axis=1)
        
        return df
    
    def features_other(self):
        df1 = self.is_weekend()
        df2 = self.near_holiday()
        
        df = pd.merge(df1, df2, on='id')
        return df
    
    def bucket_portrait(self):
        # 年龄层特征画像
        df1 = self.age_bucket_booking_rate()
        df2 = self.most_popular_place()
        df3 = self.least_popular_place()
        df4 = self.bucket_place_booking_rate()
        df5 = self.bucket_booking_place_max_num()
        df6 = self.bucket_booking_place_min_num()
        df7 = self.bucket_no_booking_rate()
        df8 = self.bucket_dest_gender_rate()
        df9 = self.bucket_dest_population()
        df10 = self.bucket_booking_avg_time()
        df11 = self.bucket_intraday_booking_rate()
        df12 = self.bucket_dest_avg_dist()
        
        ft_lst = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12]
        for ele in ft_lst:
            ele.reset_index(inplace=True, drop=True)
            ele.set_index(['age'], inplace=True)
        df = pd.concat(ft_lst, axis=1)
        df.reset_index(inplace=True)
        df.rename(columns={'index':'age'}, inplace=True)
        
        df.most_popular_place.fillna(df.most_popular_place.mode().iloc[0], inplace=True)
        df.least_popular_place.fillna(df.least_popular_place.mode().iloc[0], inplace=True)
        return df
    
    def student_portrait(self):
        df1 = self.stu_booking_rate()
        df2 = self.stu_dest_booking_rate()
        df = pd.merge(df1, df2, on='is_student')
        
        df = self.normalize(df)
        return df
    
    def season_portrait(self):
        df1 = self.season_dest_booking_rate()
        df1 = self.normalize(df1)
        return df1
    
    def month_portrait(self):
        df1 = self.month_dest_booking_rate()
        df1 = self.normalize(df1)
        return df1
    
    def extract_features(self):
        ''' user features '''
        ft_user = self.features_user()
  
        ''' bucket features '''
        df1 = self.features_age_bucket()
        df2 = self.bucket_portrait()
        df = pd.merge(df2, df1, on='age', how='left')
          
        # 处理缺失值
        df.fillna(df.mean(), inplace=True)
          
        # 数值型归一化
        ft_bucket = self.normalize(df)
        ft_bucket.rename(columns={'age':'age_bucket'}, inplace=True)
         
        ''' other features '''
        ft_other = self.features_other()
        
        ''' append portrait features '''
        pf1 = self.student_portrait()
        pf2 = self.season_portrait()
        pf3 = self.month_portrait()
        
        df_tmp = self.df_raw[['id', 'date_account_created']]
        df_tmp.date_account_created = pd.to_datetime(df_tmp.date_account_created)
        df_tmp['month'] = df_tmp.date_account_created.dt.month
        df_tmp['season'] = df_tmp.apply(lambda x : self.gen_season(x['month']), axis=1)
        ft_user = pd.merge(ft_user, df_tmp, on='id')
        ft_user = pd.merge(ft_user, pf1, on='is_student', how='left')
        ft_user = pd.merge(ft_user, pf2, on='season', how='left')
        ft_user = pd.merge(ft_user, pf3, on='month', how='left')
        ft_user.drop(['season', 'month', 'date_account_created'], axis=1, inplace=True)
        
        ''' 整合各部分特征 '''
        ft_final = pd.merge(ft_user, ft_bucket, on='age_bucket', how='left')
        ft_final = pd.merge(ft_final, ft_other, on='id')
        
        # one-hot encode
        dummies = ['age_bucket',
                   'frequent_used_device',
                   'most_popular_place',
                   'least_popular_place']
        for feature in dummies:
            dummy_features = pd.get_dummies(ft_final[feature], prefix=feature)
            for dummy in dummy_features:
                ft_final[dummy] = dummy_features[dummy]
            ft_final = ft_final.drop([feature], 1)
#         print(ft_final.isna().any())
        
        ''' Export all features '''
        ft_final.to_csv('all_features_%s.csv' % self.extr_type, index=False)
    
    def normalize(self, dataframe):
        for index, value in dataframe.dtypes.iteritems():
            if 'float' in str(value) or 'int' in str(value):
                minValue = dataframe[index].min()
                maxValue = dataframe[index].max()
                dataframe[index] = dataframe[index].apply(lambda x:(x - minValue) / (maxValue - minValue) if maxValue != minValue else x)
                
        return dataframe
    
    def gen_season(self, month):
        spring = range(1, 4)
        summer = range(4, 7)
        autumn = range(7, 10)
        winter = range(10, 13)
        
        if month in spring:
            return 'A'
        elif month in summer:
            return 'B'
        elif month in autumn:
            return 'C'
        elif month in winter:
            return 'D'
    
    def gen_age_bucket(self, age):
        bucket = 'NA'
        if age >= 0 and age <= 4:
            bucket = 'A'
        elif age >= 5 and age <= 9:
            bucket = 'B'
        elif age >= 10 and age <= 14:
            bucket = 'C'
        elif age >= 15 and age <= 19:
            bucket = 'D'
        elif age >= 20 and age <= 24:
            bucket = 'E'
        elif age >= 25 and age <= 29:
            bucket = 'F'
        elif age >= 30 and age <= 34:
            bucket = 'G'
        elif age >= 35 and age <= 39:
            bucket = 'H'
        elif age >= 40 and age <= 44:
            bucket = 'I'
        elif age >= 45 and age <= 49:
            bucket = 'J'
        elif age >= 50 and age <= 54:
            bucket = 'K'
        elif age >= 55 and age <= 59:
            bucket = 'L'
        elif age >= 60 and age <= 64:
            bucket = 'M'
        elif age >= 65 and age <= 69:
            bucket = 'N'
        elif age >= 70 and age <= 74:
            bucket = 'O'
        elif age >= 75 and age <= 79:
            bucket = 'P'
        elif age >= 80 and age <= 84:
            bucket = 'Q'
        elif age >= 85 and age <= 89:
            bucket = 'R'
        elif age >= 90 and age <= 94:
            bucket = 'S'
        elif age >= 95 and age <= 99:
            bucket = 'T'
        elif age >= 100:
            bucket = 'U'
        
        return bucket
    
    def preprocess_age_gender_bkts(self, df):
        df.age_bucket = df.age_bucket.apply(lambda x:'100' if x == '100+' else x)
        df.age_bucket = df.age_bucket.apply(lambda x:x.split('-')[0])
        df.age_bucket = df.age_bucket.astype('int')
        df.age_bucket = df.age_bucket.apply(lambda x:self.gen_age_bucket(x))
        return df
        
    



if __name__ == '__main__':
    
    extr = extract('C:\\scnguh\\datamining\\airbnb\\all\\', 'train')
     
    extr.extract_features()
    
    extr = extract('C:\\scnguh\\datamining\\airbnb\\all\\', 'test')
     
    extr.extract_features()
    
    
    
