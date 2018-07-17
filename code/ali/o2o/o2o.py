# -*- coding: utf-8 -*-
import pandas as pd
import csv
import matplotlib.pyplot as plt


class o2o(object):
    
    def clean(self):
        
        offline, online = [],[]
        # Processing offline data
        with open('C:\scnguh\datamining\o2o\ccf_offline_stage1_train\ccf_offline_stage1_train.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Date'] == 'null' and row['Coupon_id'] != 'null':
                    row['Type'] = 'Negative'
                elif row['Date'] != 'null' and row['Coupon_id'] == 'null':
                    row['Type'] = 'Normal'
                elif row['Date'] != 'null' and row['Coupon_id'] != 'null':
                    row['Type'] = 'Positive'
                else:
                    row['Type'] = 'NA'
                        
                offline.append(row)
        
        df = pd.DataFrame(offline)
        df.to_csv('C:\scnguh\datamining\o2o\ccf_offline_stage1_train\offline_clean.csv', encoding='utf-8')
        
        # Processing online data
        with open('C:\scnguh\datamining\o2o\ccf_online_stage1_train\ccf_online_stage1_train.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Date'] == 'null' and row['Coupon_id'] != 'null':
                    row['Type'] = 'Negative'
                elif row['Date'] != 'null' and row['Coupon_id'] == 'null':
                    row['Type'] = 'Normal'
                elif row['Date'] != 'null' and row['Coupon_id'] != 'null':
                    row['Type'] = 'Positive'
                else:
                    row['Type'] = 'NA'
                        
                online.append(row)
        
        df = pd.DataFrame(online)
        df.to_csv('C:\scnguh\datamining\o2o\ccf_online_stage1_train\online_clean.csv', encoding='utf-8')
    
    def summarize(self):
        df1 = pd.read_csv('C:\\scnguh\\datamining\\o2o\\ccf_offline_stage1_train\\offline_clean.csv')
        df2 = pd.read_csv('C:\\scnguh\\datamining\\o2o\\ccf_online_stage1_train\\online_clean.csv')
        for df in [df1]:
            print(df.describe())
            res = {}
            for i in df['Type']:
                res[i] = res.get(i, 0) + 1
            
            names, values = [], []
            for k,v in res.items():
                names.append(k)
                values.append(v)
                
            plt.bar(range(len(values)), values,color='rgb',tick_label=names)
            plt.show()
    
        
        
        
if __name__ == '__main__':
    o2o = o2o()
#     o2o.clean()
    o2o.summarize()




