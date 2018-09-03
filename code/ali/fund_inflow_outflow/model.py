# -*- coding: utf-8 -*-
'''
Created on Sep 2, 2018

@author: Eddy Hu
'''
import os
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt


class model(object):
    '''
    Use FaceBook Prophet to build the time series model
    '''
    
    def __init__(self, file):
        
        self.data = pd.read_csv(file, dtype={'total_purchase_amt':float, 'total_redeem_amt':float}, parse_dates=['report_date'])
        
        # 删除过年期间异常数据
        self.data.drop(self.data.index[213:222], inplace=True)
        
        self.data['total_purchase_amt'] = self.data.total_purchase_amt.apply(lambda x:x * 0.01)
        self.data['total_redeem_amt'] = self.data.total_redeem_amt.apply(lambda x:x * 0.01)
    
    def convert_purchase_data(self):
        
        self.df_purchase = self.data.copy()
        self.df_purchase = self.df_purchase.iloc[:, 0:2]
        self.df_purchase.rename(columns={'report_date':'ds', 'total_purchase_amt':'y'}, inplace=True)
        
        return self.df_purchase

    def convert_redeem_data(self):
        
        self.df_redeem = self.data.copy()
        self.df_redeem = self.df_redeem.iloc[:, [0, 2]]
        self.df_redeem.rename(columns={'report_date':'ds', 'total_redeem_amt':'y'}, inplace=True)
        
        return self.df_redeem

    def build_model(self, train):
        
        train['y'] = np.log(train.y)  # 降低振幅
        
        # 处理特殊十一黄金周和双十一，lower_window=-1表示昨天也是假期，upper_window=2表示明天后天也是假期
        national_day = pd.DataFrame({
            'holiday' : 'national',
            'ds' : pd.to_datetime(['2013-10-01']),
            'lower_window' : 0,
            'upper_window' : 6
        })
        
        double_11 = pd.DataFrame({
            'holiday' : 'd11',
            'ds' : pd.to_datetime(['2013-11-11']),
            'lower_window' :-10,
            'upper_window' : 5
        })
        
        tomb_sweep = pd.DataFrame({
            'holiday' : 'ts',
            'ds' : pd.to_datetime(['2014-04-05']),
            'lower_window' : 0,
            'upper_window' : 2
        })
        
        mid_autumn = pd.DataFrame({
            'holiday' : 'ma',
            'ds' : pd.to_datetime(['2013-09-19']),
            'lower_window' : 0,
            'upper_window' : 2
        })
        
        holidays = pd.concat([national_day, double_11, tomb_sweep, mid_autumn])
        
        m = Prophet(holidays=holidays,
                    interval_width=0.95,  # 希望有一些不确定性的变化而不是和历史趋势一致
                    mcmc_samples=300)  # 为获得季节的不确定性需要进行完整贝叶斯抽样
        
        # 处理季度效应
        m.add_seasonality(name='quarterly', period=120, fourier_order=5)
        
        # 处理月效应 ，增加傅里叶项数可以适应更快的周期变化但容易过拟合
        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # 处理周效应，prior_scale为正则化强度
        m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)
        
        m.fit(train)
        
#         self.cross_validate(m, '100 days', save=True)
        
        future = m.make_future_dataframe(periods=30, freq='D')
        forecast = m.predict(future)
        
        m.plot_components(forecast)
        plt.savefig('plot_components.jpg')
        
        # log 逆运算
        df = forecast[['ds', 'yhat']]
        df['yhat'] = df.yhat.apply(lambda x:np.exp(x))
        
        return df
    
    def cross_validate(self, model, horizon, save=False):
        
        # 交叉检验
        from fbprophet.diagnostics import cross_validation
        
        df_cv = cross_validation(model, horizon=horizon)
        if save is True:
            df_cv.to_csv('c:\\scnguh\\cross_validate.csv', index=False)
        else:
            return df_cv

    def generate_forecast(self):
        
        res_purchase = self.build_model(self.convert_purchase_data())
        res_purchase.rename(columns={'yhat':'purchase'}, inplace=True)
        
        res_redeem = self.build_model(self.convert_redeem_data())
        res_redeem.rename(columns={'yhat':'redeem'}, inplace=True)
        
        result = pd.merge(res_purchase, res_redeem, on=['ds'])
        
        # 格式化输出
        result['purchase'] = result.purchase.apply(lambda x:'%.2f' % x)
        result['redeem'] = result.redeem.apply(lambda x:'%.2f' % x)
        result.rename(columns={'ds':'report_date'}, inplace=True)
        result['purchase'] = result.purchase.apply(lambda x:x.replace('.', ''))
        result['redeem'] = result.redeem.apply(lambda x:x.replace('.', ''))
        
        return result.iloc[418:448, :]
        
        



if __name__ == '__main__':
    
    train_ori = 'C:\\scnguh\\datamining\\fund_inflow_outflow\\Purchase&Redemption Data' \
                '\\Purchase&Redemption Data\\temp\\day_purchase_redeem.csv'
    
    mo = model(train_ori)
    
    res = mo.generate_forecast()
    
    res.to_csv('forecast.csv', index=False)
    
    
    
    
    
    
    
    
    
