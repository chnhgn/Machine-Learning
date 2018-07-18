# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import dateutil


class split(object):
    
    def __init__(self, online, offline, target_dir):
        self.online = online
        self.offline = offline
        self.target_dir = os.path.join(target_dir, 'temp')
        
    def split(self, ds_name, slice_date):
        # Create the target directory
        if os.path.exists(self.target_dir) is not True:
            os.mkdir(self.target_dir)
        
        if ds_name == 'offline':
            ds = self.offline
        if ds_name == 'online':
            ds = self.online
            
        df = pd.read_csv(ds, dtype=str, keep_default_na=False)
#         print(df[(df.Date_received == 'null') & (df.Date == 'null')])
        df['temp_date'] = 0     # Split point
        
        df1 = df[(df.Date_received != 'null') & (df.Date != 'null')]
        df1['temp_date'] = df1['Date_received']
        
        df2 = df[df.Date_received == 'null']
        df2['temp_date'] = df2['Date']
        
        df = pd.concat([df1, df2])
        
        df['temp_date'] = pd.to_datetime(df.temp_date)
#         print(df.dtypes)
        df.set_index(['temp_date'], inplace=True)
        df.sort_values(by='temp_date', inplace=True)
        delay_day = pd.to_datetime(slice_date) + dateutil.relativedelta.relativedelta(days=1)
        
        part1 = df.truncate(after=slice_date, copy=True)
        part2 = df.truncate(before=delay_day, copy=True)
                
        part1.to_csv(os.path.join(self.target_dir, ds_name+'_part1.csv'), encoding='utf-8')
        part2.to_csv(os.path.join(self.target_dir, ds_name+'_part2.csv'), encoding='utf-8')

    def remove_dup(self):
        files = os.listdir(self.target_dir)
        for f in files:
            df = pd.read_csv(os.path.join(self.target_dir, f), dtype=str, keep_default_na=False)
            before_lines = len(df)     # Line number before removing
            if 'temp_date' in df.columns:
                del df['temp_date']
                
            df.drop_duplicates(inplace=True)
            after_lines = len(df)
            drop_lines = before_lines - after_lines
            print('Removed the duplicated lines %s in %s' % (str(drop_lines), f))
            
            df.to_csv(os.path.join(self.target_dir, f), encoding='utf-8')
            
    



if __name__ == '__main__':
    
    sp = split('C:\\scnguh\\datamining\\o2o\\ccf_online_stage1_train\\ccf_online_stage1_train.csv',
              'C:\\scnguh\\datamining\\o2o\\ccf_offline_stage1_train\\ccf_offline_stage1_train.csv',
              'C:\\scnguh\\datamining\\o2o')
    
    sp.split('offline', '2016-4-15')
    sp.split('online', '2016-4-15')
    
    sp.remove_dup()
    
    
    
    
    
    
    
    
    
    
    
    
    
