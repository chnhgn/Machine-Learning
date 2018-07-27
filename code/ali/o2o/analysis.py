# -*- coding: utf-8 -*-
import os
import pandas as pd
import scipy.stats as stats
import numpy as np


class analysis(object):
    
    def __init__(self, data_path):
        self.feature_data_dir = os.path.join(data_path, 'tmp_features')
        
        self.train = pd.read_csv(os.path.join(self.feature_data_dir, 'o2o_train.csv'),
                            dtype=str,
                            keep_default_na=False)
    
    def correlation(self):
        """
        Dropped features:
            online_action_times
            online_hit_action_rate
            online_buy_action_rate
            online_get_cp_action_rate
            online_no_consume_num
            online_cp_used_num
            online_cp_used_rate
            offline_cp_used_num_in_online_and_offline
            offline_cp_record_in_online_and_offline
            user_merchant_cp_unused_num_outof_merchant_unused_num
            picked_cp_num_from_merchant
            cp_not_used_num_in_merchant
            cp_used_num_in_merchant
            cp_used_discount_mean_in_merchant
            cp_used_discount_max_in_merchant
            cp_used_types_in_merchant
            cp_used_types_outof_all_types_in_merchant
            cp_used_max_distance_in_merchant
            cp_used_min_distance_in_merchant
        """
        columns = ['Discount_rate', 'Distance', 'Date_received', 'get_cp_num', 'get_cp_not_use', 'get_cp_used', 'used_rate',
                   'x:y_cp_used_rate', 'x:y_in_all_cp_used_rate', 'cp_used_discount_mean', 'cp_used_discount_max', 'cp_used_discount_min',
                   'cp_used_merchants_outof_all_merchants', 'get_cp_used_rate', 'cp_used_per_shop', 'distance_max', 'distance_min', 'distance_mean',
                   'offline_no_consume_num_in_online_and_offline', 'user_cp_not_use_num_in_merchant', 'user_cp_used_num_in_merchant',
                   'user_get_cp_num_in_merchant', 'user_cp_used_rate_in_merchant', 'user_merchant_cp_unused_num_outof_all_user_unused_num',
                   'user_merchant_cp_used_num_outof_all_user_used_num', 'user_merchant_cp_used_num_outof_merchant_unused_num', 'cp_used_discount_min_in_merchant',
                   'cp_used_num_per_user_in_merchant', 'cp_used_num_per_type_in_merchant', 'cp_used_duration_in_merchant', 'cp_used_mean_distance_in_merchant',
                   'cp_used_rate_in_merchant', 'cp_type', 'cp_discount', 'min_consume', 'cp_freq', 'cp_total_used_num', 'cp_total_used_rate', 'cp_used_duration_total_mean',
                   'cp_total_received_num']
        
        df_corr = pd.DataFrame(columns=['v1', 'v2', 'r_value', 'p_value'])
        
        for col in columns:
            self.train[col] = eval("self.train['%s'].astype('float')" % col)
        
        for col1 in columns:
            for col2 in columns:
                x = self.train[col1].tolist()
                y = self.train[col2].tolist()
#                 x = np.array(x).astype(np.float)
#                 y = np.array(y).astype(np.float)
                r, p = stats.pearsonr(x, y)
                df_corr = df_corr.append({'v1':col1, 'v2':col2, 'r_value':r, 'p_value':p}, ignore_index=True)
        
        alpha = 0.05
        print(df_corr[(df_corr.p_value > alpha)])





if __name__ == '__main__':
    ana = analysis('C:\\scnguh\\datamining\\o2o')
    
    ana.correlation()
    
    
