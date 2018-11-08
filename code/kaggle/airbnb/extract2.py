# -*- coding: utf-8 -*-
'''
Created on Nov 6, 2018

@author: Eddy Hu
'''

import pandas as pd
pd.set_option('display.max_columns', None)
import datetime
import numpy as np



raw_path = 'C:\\scnguh\\datamining\\airbnb\\all\\'
raw_train = raw_path + 'train_clean.csv'
raw_test = raw_path + 'test_clean.csv'
raw_session = raw_path + 'sessions.csv'
raw_countries = raw_path + 'countries.csv'
raw_bucket = raw_path + 'age_gender_bkts.csv'

df_train = pd.read_csv(raw_train)
df_test = pd.read_csv(raw_test)
df_session = pd.read_csv(raw_session, index_col=False)
df_countries = pd.read_csv(raw_countries, index_col=False)
df_bucket = pd.read_csv(raw_bucket, index_col=False)

df_all = pd.concat([df_train, df_test])


# Home made One Hot Encoding function
def convert_to_binary(df, column_to_convert):
    for feature in column_to_convert:
        dummy_features = pd.get_dummies(df[feature], prefix=feature)
        for dummy in dummy_features:
            df[dummy] = dummy_features[dummy]
        df.drop([feature], 1, inplace=True)

    return df

# One Hot Encoding
columns_to_convert = ['gender',
                      'signup_method',
                      'signup_flow',
                      'language',
                      'affiliate_channel',
                      'affiliate_provider',
                      'first_affiliate_tracked',
                      'signup_app',
                      'first_device_type',
                      'first_browser']

df_all = convert_to_binary(df_all, columns_to_convert)

# Add new date related fields
df_all.date_account_created = pd.to_datetime(df_all.date_account_created)
df_all.timestamp_first_active = pd.to_datetime(df_all.timestamp_first_active)
df_all['day_account_created'] = df_all['date_account_created'].dt.weekday
df_all['month_account_created'] = df_all['date_account_created'].dt.month
df_all['quarter_account_created'] = df_all['date_account_created'].dt.quarter
df_all['year_account_created'] = df_all['date_account_created'].dt.year
df_all['hour_first_active'] = df_all['timestamp_first_active'].dt.hour
df_all['day_first_active'] = df_all['timestamp_first_active'].dt.weekday
df_all['month_first_active'] = df_all['timestamp_first_active'].dt.month
df_all['quarter_first_active'] = df_all['timestamp_first_active'].dt.quarter
df_all['year_first_active'] = df_all['timestamp_first_active'].dt.year
df_all['account_created_delay'] = (df_all['date_account_created'] - df_all['timestamp_first_active']).dt.days

# Drop unnecessary columns
columns_to_drop = ['date_account_created', 'timestamp_first_active', 'country_destination']
df_all.drop(columns_to_drop, axis=1, inplace=True)
        
''' 
    Merge age bucket data 
'''      
def gen_age_bucket(age):
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

df_bucket.age_bucket = df_bucket.age_bucket.apply(lambda x:'100' if x == '100+' else x)
df_bucket.age_bucket = df_bucket.age_bucket.apply(lambda x:x.split('-')[0])
df_bucket.age_bucket = df_bucket.age_bucket.astype('int')
df_bucket.age_bucket = df_bucket.age_bucket.apply(lambda x:gen_age_bucket(x))
df_bucket.rename(columns={'age_bucket' : 'age'}, inplace=True)

df_all.age = df_all.age.apply(lambda x : gen_age_bucket(x))

# Country population
bkt = df_bucket.groupby(['age', 'country_destination'], as_index=False).agg({'population_in_thousands' : 'sum'})
bkt.reset_index(inplace=True, drop=True)
bkt = bkt.pivot('age', 'country_destination', 'population_in_thousands')
bkt.reset_index(inplace=True)
for index, value in bkt.dtypes.iteritems():
    if index != 'age':
        bkt.rename(columns={index:index + '_population'}, inplace=True)

# Merge population to df_all
df_all = pd.merge(df_all, bkt, on='age', how='left')

# Gender population
bkt = df_bucket.groupby(['age', 'gender'], as_index=False).agg({'population_in_thousands' : 'sum'})
bkt.reset_index(inplace=True, drop=True)
bkt = bkt.pivot('age', 'gender', 'population_in_thousands')
bkt.reset_index(inplace=True)
for index, value in bkt.dtypes.iteritems():
    if index != 'age':
        bkt.rename(columns={index:index + '_population'}, inplace=True)

# Merge population to df_all
df_all = pd.merge(df_all, bkt, on='age', how='left')
df_all = convert_to_binary(df_all, ['age'])
df_all.fillna(-1, inplace=True)




'''
    Merge session data
'''
# Process null value
df_session.action.fillna('NaN', inplace=True)
df_session.action_type.fillna('NaN', inplace=True)
df_session.action_detail.fillna('NaN', inplace=True)
df_session.device_type.fillna('NaN', inplace=True)
df_session.secs_elapsed.fillna(0, inplace=True)
df_session.rename(columns={'user_id' : 'id'}, inplace=True)

columns_sess = ['action', 'action_type', 'action_detail', 'device_type']
for col in columns_sess:
    sess = df_session.groupby(['id', '%s' % col], as_index=False).agg({'secs_elapsed' : 'sum'})
    sess = sess.pivot('id', '%s' % col, 'secs_elapsed')
    sess.reset_index(inplace=True)
    for index, value in sess.dtypes.iteritems():
        if index != 'id':
            sess.rename(columns={index:index + '_%s' % col}, inplace=True)
    sess.fillna(-1, inplace=True)
    df_all = pd.merge(df_all, sess, on='id', how='left')

df_all.fillna(-1, inplace=True)




'''
    Normalize data
'''
for index, value in df_all.dtypes.iteritems():
    if index != 'id':
        minValue = df_all[index].min()
        maxValue = df_all[index].max()
        df_all[index] = df_all[index].apply(lambda x:(x - minValue) / (maxValue - minValue))




'''
    Split data and save
'''
train = df_all[:-len(df_test)]
test = df_all[-len(df_test):]

# Add back label
train = pd.merge(train, df_train[['id', 'country_destination']], on='id')

# Save data
train.to_csv('./data/train.csv', index=False)
test.to_csv('./data/test.csv', index=False)
print('Save finished!')

