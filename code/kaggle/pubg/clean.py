# -*- coding: utf-8 -*-
'''
Created on Nov 13, 2018

@author: Eddy Hu
'''

import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)


data_dir = "C:\\scnguh\\datamining\\pubg\\all\\"

train = pd.read_csv(data_dir + "train_V2.csv")
test = pd.read_csv(data_dir + "test_V2.csv")

# killPlace
train = train[train.killPlace <= 100]

# winPlacePerc should not be null
train = train[train.winPlacePerc >= 0 ]

df_all = pd.concat([train, test])

# Split train/test
clean_train = df_all[:-len(test)]
clean_test = df_all[-len(test):]

clean_train.to_csv("./data/train.csv", index=False)
clean_test.to_csv("./data/test.csv", index=False)
print("Saving completed!")



