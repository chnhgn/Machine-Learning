# -*- coding: utf-8 -*-
'''
Created on Aug 14, 2018

@author: Eddy Hu
'''

import pandas as pd
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA
from dateutil.parser import parse



class time_series(object):
    
    def __init__(self):
        
        repo = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
        data = pd.read_csv(os.path.join(repo, 'ibm-common-stock-closing-prices.csv'), index_col='Date')
        data.index = pd.to_datetime(data.index)
        self.data = data.T.squeeze()    # to series
    
    def trend_plot(self):    
        plt.plot(self.data['Date'], self.data['Prices'])
        plt.show()
        
    def draw_acf_pacf(self, ts, lags=31):
        ''' 自相关和偏相关图，默认阶数为31阶 '''
        f = plt.figure(facecolor='white')
        ax1 = f.add_subplot(211)
        plot_acf(ts, lags=31, ax=ax1)
        ax2 = f.add_subplot(212)
        plot_pacf(ts, lags=31, ax=ax2)
        plt.show()
    
    def draw_ts(self, ts):
        plt.figure(facecolor='white')
        ts.plot(color='blue')
        plt.show()
    
    def draw_moving(self, timeSeries, size):
        ''' 移动平均图 '''
        df = timeSeries.to_frame()
        f = plt.figure(facecolor='white')
        # 对size个数据进行移动平均
        rol_mean = df.rolling(window=size).mean()
        timeSeries.plot(color='blue', label='Original')
        rol_mean.plot(color='red', label='Rolling Mean')
        plt.legend(loc='best')
        plt.title('Rolling Mean')
        plt.show()
    
    def test_stationarity(self, ts):
        dftest = adfuller(ts)
        # 对上述函数求得的值进行语义描述
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        return dfoutput
    
    def difference(self, dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return np.array(diff)

    def inverse_difference(self, history, yhat, interval=1):
        return yhat + history[-interval]
    
    
    def inverse_moving_mean(self, history, yhat, interval):
        return yhat*12 - sum(history[-(interval-1):])
    
        


if __name__ == '__main__':
    
    ts = time_series()
    ts_ori = ts.data
    
    # 处理振幅
    ts_log = np.log(ts.data)                        
    
    # 移动均值处理长期趋势
    rol_mean = ts_log.to_frame().rolling(window=12).mean()
    rol_mean.dropna(inplace=True)
    rol_mean = rol_mean.T.squeeze()
    
    # 一阶差分处理周期因素
    diff_1 = rol_mean.diff(1)                         
    diff_1.dropna(inplace=True)
    print(ts.test_stationarity(diff_1))
    print(diff_1.describe())
    
#     ts.draw_acf_pacf(diff_1)    # 自相关拖尾，偏相关拖尾，模型识别为ARMA
    model = ARMA(diff_1, order=(1, 1))
    result_arma = model.fit( disp=-1, method='css')
    predict_ts = result_arma.predict()
    
    # 相关逆变换
    diff_shift_ts = rol_mean.shift(1)     # 一阶差分还原
    diff_recover_1 = predict_ts.add(diff_shift_ts)
    
    rol_sum = ts_log.rolling(window=11).sum()   # 移动平均还原
    rol_recover = diff_recover_1*12 - rol_sum.shift(1)
    
    log_recover = np.exp(rol_recover)   # 对数还原
    log_recover.dropna(inplace=True)
    
    # 均方根误差检验拟合程度
    ts_ori = ts_ori[log_recover.index]      # 解决两个比较序列数量不一致问题
    plt.figure(facecolor='white')
    log_recover.plot(color='blue', label='Predict')
    ts_ori.plot(color='red', label='Original')
    plt.legend(loc='best')
    plt.title('RMSE: %.4f'% np.sqrt(sum((log_recover-ts_ori)**2)/ts_ori.size))
    plt.grid()
    plt.show()
    
    # 预测
    forecast_ts = result_arma.predict(start=len(diff_1), end=len(diff_1)+6)
    # 以第一个预测值为例子进行逆运算
    inverse_diff = ts.inverse_difference(rol_mean.tolist(), forecast_ts[996], 1)
    inverse_moving_mean = ts.inverse_moving_mean(ts_log.tolist(), inverse_diff, 12)
    inverse_log = np.exp(inverse_moving_mean)
    print(inverse_log)
    
    
    
    
    