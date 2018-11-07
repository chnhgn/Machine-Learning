# -*- coding: utf-8 -*-
'''
Created on Nov 6, 2018

@author: Eddy Hu
'''

import pandas as pd
pd.set_option('display.max_columns', None)
import datetime
import numpy as np



def gen_age_bucket(age):
    bucket = np.NaN
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

def normalize(dataframe):
    for index, value in dataframe.dtypes.iteritems():
        if str(value) == 'float64':
            minValue = dataframe[index].min()
            maxValue = dataframe[index].max()
            dataframe[index] = dataframe[index].apply(lambda x:(x - minValue) / (maxValue - minValue) if maxValue != minValue else x)
            
    return dataframe

def preprocess_age_gender_bkts(df):
    df.age_bucket = df.age_bucket.apply(lambda x:'100' if x == '100+' else x)
    df.age_bucket = df.age_bucket.apply(lambda x:x.split('-')[0])
    df.age_bucket = df.age_bucket.astype('int')
    df.age_bucket = df.age_bucket.apply(lambda x:gen_age_bucket(x))
    return df

data_dir = 'C:\\scnguh\\datamining\\airbnb\\all\\'

df_train = pd.read_csv(data_dir + 'train_users.csv')
df_test = pd.read_csv(data_dir + 'test_users.csv')
df_age_bkt = preprocess_age_gender_bkts(pd.read_csv(data_dir + 'age_gender_bkts.csv'))
df_countries = pd.read_csv(data_dir + 'countries.csv')
df_session = pd.read_csv(data_dir + 'sessions.csv')



''' Merge train and test data '''
test_size = len(df_test)
df_head = pd.concat([df_train, df_test])
df_head.date_account_created = pd.to_datetime(df_head.date_account_created)
df_head.date_first_booking = pd.to_datetime(df_head.date_first_booking)
df_head.signup_flow = df_head.signup_flow.astype('str')
df_head.timestamp_first_active = df_head.timestamp_first_active.astype('str')
df_head.timestamp_first_active = df_head.timestamp_first_active.apply(lambda x:datetime.datetime.strptime(x, '%Y%m%d%H%M%S').date())
df_head.timestamp_first_active = pd.to_datetime(df_head.timestamp_first_active)

df_head.age = df_head.age.apply(lambda x : gen_age_bucket(x))
df_head['account_created_delay'] = df_head.date_account_created - df_head.timestamp_first_active
df_head['first_booking_delay'] = df_head.date_first_booking - df_head.timestamp_first_active
df_head.account_created_delay = df_head.account_created_delay.apply(lambda x : float(x.days))
df_head.first_booking_delay = df_head.first_booking_delay.apply(lambda x : float(x.days))
df_head.drop(['date_account_created', 'date_first_booking', 'timestamp_first_active'], axis=1, inplace=True)

# Encode nominal column with one-hot code
for index, value in df_head.dtypes.iteritems():
    if index not in ['id', 'country_destination', 'age']:
        if value == 'object':
            dummy_features = pd.get_dummies(df_head[index], prefix=index, dummy_na=True)
            for dummy in dummy_features:
                df_head[dummy] = dummy_features[dummy]
            df_head.drop([index], 1, inplace=True)
            
''' Merge countries data '''
dummy_features = pd.get_dummies(df_countries.destination_language, prefix='destination_language', dummy_na=True)
for dummy in dummy_features:
    df_countries[dummy] = dummy_features[dummy]
df_countries.drop(['destination_language'], 1, inplace=True)

df_head = pd.merge(df_head, df_countries, on='country_destination', how='left')

''' Merge age bucket data '''
df_age_bkt.rename(columns={'age_bucket' : 'age'}, inplace=True)
df_age_bkt = df_age_bkt.groupby(['age', 'country_destination'], as_index=False).agg({'population_in_thousands' : 'sum'})
df_age_bkt.reset_index(inplace=True, drop=True)

df_head = pd.merge(df_head, df_age_bkt, on=['age', 'country_destination'], how='left')
# One-hot age
dummy_features = pd.get_dummies(df_head.age, prefix='age', dummy_na=True)
for dummy in dummy_features:
    df_head[dummy] = dummy_features[dummy]
df_head.drop(['age'], 1, inplace=True)

''' Merge session data '''
df1 = df_session[['user_id', 'action', 'secs_elapsed']]
df1.action.fillna('nan', inplace=True)
df1 = df1.groupby(['user_id', 'action'], as_index=False).agg({'secs_elapsed' : 'sum'})
df1 = df1.pivot('user_id', 'action', 'secs_elapsed')
df1.reset_index(inplace=True)
df1.set_index(['user_id'], inplace=True)
for index, value in df1.dtypes.iteritems():
    if index != 'user_id':
        df1.rename(columns={index:'action_' + index}, inplace=True)

df2 = df_session[['user_id', 'action_type', 'secs_elapsed']]
df2.action_type.fillna('nan', inplace=True)
df2 = df2.groupby(['user_id', 'action_type'], as_index=False).agg({'secs_elapsed' : 'sum'})
df2 = df2.pivot('user_id', 'action_type', 'secs_elapsed')
df2.reset_index(inplace=True)
df2.set_index(['user_id'], inplace=True)
for index, value in df2.dtypes.iteritems():
    if index != 'user_id':
        df2.rename(columns={index:'action_type_' + index}, inplace=True)

df3 = df_session[['user_id', 'action_detail', 'secs_elapsed']]
df3.action_detail.fillna('nan', inplace=True)
df3 = df3.groupby(['user_id', 'action_detail'], as_index=False).agg({'secs_elapsed' : 'sum'})
df3 = df3.pivot('user_id', 'action_detail', 'secs_elapsed')
df3.reset_index(inplace=True)
df3.set_index(['user_id'], inplace=True)
for index, value in df3.dtypes.iteritems():
    if index != 'user_id':
        df3.rename(columns={index:'action_detail_' + index}, inplace=True)

df4 = df_session[['user_id', 'device_type', 'secs_elapsed']]
df4.device_type.fillna('nan', inplace=True)
df4 = df4.groupby(['user_id', 'device_type'], as_index=False).agg({'secs_elapsed' : 'sum'})
df4 = df4.pivot('user_id', 'device_type', 'secs_elapsed')
df4.reset_index(inplace=True)
df4.set_index(['user_id'], inplace=True)
for index, value in df4.dtypes.iteritems():
    if index != 'user_id':
        df4.rename(columns={index:'device_type_' + index}, inplace=True)

df_tail = pd.concat([df1, df2, df3, df4], axis=1)      
df_tail.reset_index(inplace=True)
df_tail.rename(columns={'user_id':'id'}, inplace=True)
        
''' Merge all features '''
df = pd.merge(df_head, df_tail, on='id', how='left')

''' Normalize data '''
df = normalize(df)

''' Decompose data to train/test '''       
train = df[:-test_size]
test = df[-test_size:]

''' Save data '''
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)


