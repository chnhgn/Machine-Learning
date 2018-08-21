# -*- coding: utf-8 -*-
'''
Created on Aug 21, 2018

@author: Eddy Hu
'''
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller


class extract(object):
    
    def __init__(self, directory):
        
        self.user_balance = pd.read_csv(os.path.join(directory, 'user_balance_table.csv'), dtype=str)
        
        self.temp_dir = os.path.join(directory, 'temp')
        
        if os.path.exists(self.temp_dir) is not True:
            os.mkdir(self.temp_dir)
    
    def calc_redeem_purchase(self):
        
        df = self.user_balance
        df['total_purchase_amt'] = df.total_purchase_amt.astype('int')
        df['total_redeem_amt'] = df.total_redeem_amt.astype('int')
        df['report_date'] = pd.to_datetime(df.report_date, format='%Y-%m-%d')
        df_0 = df[(df.total_purchase_amt > 0)&(df.total_redeem_amt > 0)][['user_id', 'report_date', 'total_purchase_amt', 'total_redeem_amt']]
        df_0.sort_values(by='report_date', inplace=True)
        
        df_purchase = df_0.groupby(['report_date'], as_index=False)['total_purchase_amt'].sum()
        df_redeem = df_0.groupby(['report_date'], as_index=False)['total_redeem_amt'].sum()
        
        df_total = pd.merge(df_purchase, df_redeem, on='report_date')
        return df_total
    
    def trend_plot(self, df):
        f = plt.figure(facecolor='white')
        ax1 = f.add_subplot(211)
        df.plot(x='report_date', y='total_purchase_amt', ax=ax1)
        
        ax2 = f.add_subplot(212)
        df.plot(x='report_date', y='total_redeem_amt', ax=ax2)
        
        plt.show()
    
    def test_stationarity(self, ts):
        dftest = adfuller(ts)
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        return dfoutput
    
    



if __name__ == '__main__':
    
    ext = extract('C:\\scnguh\\datamining\\fund_inflow_outflow\\Purchase&Redemption Data\\Purchase&Redemption Data')
    
    df = ext.calc_redeem_purchase()
    
#     ext.trend_plot(df)
    
    series1 = df.iloc[:,[0,1]]
    series2 = df.iloc[:,[0,2]]
    series1.set_index(['report_date'], inplace=True)
    series2.set_index(['report_date'], inplace=True)
    series1 = series1.T.squeeze()
    series2 = series2.T.squeeze()
    
    print(ext.test_stationarity(series1))
    
    print(ext.test_stationarity(series2))
    
    
    
    
    
    