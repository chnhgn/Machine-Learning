# -*- coding: utf-8 -*-
import pandas as pd
import os
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt

raw_path = 'C:\\scnguh\\datamining\\airbnb\\all\\'
raw_train = raw_path + 'train_users.csv'
raw_test = raw_path + 'test_users.csv'
raw_session = raw_path + 'sessions.csv'

if os.path.exists('data') is False:
    os.mkdir('data')
    
df_raw_train = pd.read_csv(raw_train)
df_raw_test = pd.read_csv(raw_test)
df_raw_session = pd.read_csv(raw_session)

# Merge train/test data to process
df_all = pd.concat([df_raw_train, df_raw_test])

df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'], format='%Y-%m-%d')
df_all['timestamp_first_active'] = pd.to_datetime(df_all['timestamp_first_active'], format='%Y%m%d%H%M%S')
df_all['date_account_created'].fillna(df_all.timestamp_first_active, inplace=True)

df_all.drop('date_first_booking', axis=1, inplace=True)

# Remove outliers function
def remove_outliers(df, column, min_val, max_val):
    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values <= min_val, col_values >= max_val), np.NaN, col_values)
    return df

# Fixing age column
df_all = remove_outliers(df=df_all, column='age', min_val=15, max_val=90)
df_all['age'].fillna(-1, inplace=True)
df_all['first_affiliate_tracked'].fillna(-1, inplace=True)

# Split the train/test
train = df_all[:-len(df_raw_test)]
test = df_all[-len(df_raw_test):]

train.to_csv(raw_path + 'train_clean.csv', index=False)
test.to_csv(raw_path + 'test_clean.csv', index=False)

