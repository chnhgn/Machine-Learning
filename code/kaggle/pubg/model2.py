# -*- coding: utf-8 -*-
'''
Created on Nov 21, 2018

@author: Eddy Hu
'''
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option("display.max_columns", None)
import xgboost as xgb
from xgboost.sklearn import XGBRegressor


data_dir = "C:\\scnguh\\datamining\\pubg\\all\\"

train = pd.read_csv(data_dir + "train_V2.csv")
test = pd.read_csv(data_dir + "test_V2.csv")

# killPlace
train = train[train.killPlace <= 100]
# winPlacePerc should not be null
train = train[train.winPlacePerc >= 0 ]

df_all = pd.concat([train, test])

# # Feature project, part1 Personal features

# DBNOs ratio in whole team
df1 = df_all.groupby(['groupId'], as_index=False).agg({'DBNOs':'sum'})
df1.reset_index(inplace=True, drop=True)
df1.rename(columns={'DBNOs':'DBNOs_team'}, inplace=True)
df2 = pd.merge(df_all, df1, on='groupId', how='left')
df2 = df2[['Id', 'DBNOs', 'DBNOs_team']]
df2.loc[:, 'DBNOs_ratio'] = df2.DBNOs / df2.DBNOs_team
feature1 = df2[['Id', 'DBNOs_ratio']]
feature1.fillna(0, inplace=True)

# Assists ratio in whole team
df1 = df_all.groupby(['groupId'], as_index=False).agg({'assists':'sum'})
df1.reset_index(inplace=True, drop=True)
df1.rename(columns={'assists':'assists_team'}, inplace=True)
df2 = pd.merge(df_all, df1, on='groupId', how='left')
df2 = df2[['Id', 'assists', 'assists_team']]
df2.loc[:, 'assists_ratio'] = df2.assists / df2.assists_team
feature2 = df2[['Id', 'assists_ratio']]
feature2.fillna(0, inplace=True)

# Damage ratio in whole team
df1 = df_all.groupby(['groupId'], as_index=False).agg({'damageDealt':'sum'})
df1.reset_index(inplace=True, drop=True)
df1.rename(columns={'damageDealt':'damageDealt_team'}, inplace=True)
df2 = pd.merge(df_all, df1, on='groupId', how='left')
df2 = df2[['Id', 'damageDealt', 'damageDealt_team']]
df2.loc[:, 'damageDealt_ratio'] = df2.damageDealt / df2.damageDealt_team
feature3 = df2[['Id', 'damageDealt_ratio']]
feature3.fillna(0, inplace=True)

# Headshot ratio
df1 = df_all[['Id', 'headshotKills', 'kills']]
df1.loc[:, 'headshot_ratio'] = df1.headshotKills / df1.kills
feature4 = df1[['Id', 'headshot_ratio']]
feature4.fillna(0, inplace=True)

# Heals ratio in whole team
df1 = df_all.groupby(['groupId'], as_index=False).agg({'heals':'sum'})
df1.reset_index(inplace=True, drop=True)
df1.rename(columns={'heals':'heals_team'}, inplace=True)
df2 = pd.merge(df_all, df1, on='groupId', how='left')
df2 = df2[['Id', 'heals', 'heals_team']]
df2.loc[:, 'heals_ratio'] = df2.heals / df2.heals_team
feature5 = df2[['Id', 'heals_ratio']]
feature5.fillna(0, inplace=True)

# Rank delta
df1 = df_all[['Id', 'killPlace', 'maxPlace']]
df1.loc[:, 'rank_delta'] = abs(df1.killPlace - df1.maxPlace)
feature6 = df1[['Id', 'rank_delta']]

# Is kill place top 10
df1 = df_all[['Id', 'killPlace']]
df2 = df1[df1.killPlace <= 10]
df3 = df1[df1.killPlace > 10]
df2.loc[:, 'is_killplace_top10'] = 1
df3.loc[:, 'is_killplace_top10'] = 0
df4 = pd.concat([df2, df3])
feature7 = df4[['Id', 'is_killplace_top10']]

# Total moving distance
df1 = df_all[['Id', 'rideDistance', 'swimDistance', 'walkDistance']]
df1.loc[:, 'total_distance'] = df1.rideDistance + df1.swimDistance + df1.walkDistance
feature8 = df1[['Id', 'total_distance']]

# Kills in match max kills
df1 = df_all[['Id', 'matchId', 'kills']]
df2 = df1.groupby(['matchId'], as_index=False).agg({'kills':'max'})
df2.rename(columns={'kills':'max_kills'}, inplace=True)
df3 = pd.merge(df1, df2, on='matchId', how='left')
df3.loc[:, 'kills_in_max_kills'] = df3.kills / df3.max_kills
feature9 = df3[['Id', 'kills_in_max_kills']]
feature9.fillna(0, inplace=True)

# Damage in match max damages
df1 = df_all[['Id', 'matchId', 'damageDealt']]
df2 = df1.groupby(['matchId'], as_index=False).agg({'damageDealt':'max'})
df2.rename(columns={'damageDealt':'max_damage'}, inplace=True)
df3 = pd.merge(df1, df2, on='matchId', how='left')
df3.loc[:, 'damage_in_max_damages'] = df3.damageDealt / df3.max_damage
feature10 = df3[['Id', 'damage_in_max_damages']]
feature10.fillna(0, inplace=True)

# Revive in match max revives
df1 = df_all[['Id', 'matchId', 'revives']]
df2 = df1.groupby(['matchId'], as_index=False).agg({'revives':'max'})
df2.rename(columns={'revives':'max_revive'}, inplace=True)
df3 = pd.merge(df1, df2, on='matchId', how='left')
df3.loc[:, 'revive_in_max_revives'] = df3.revives / df3.max_revive
feature11 = df3[['Id', 'revive_in_max_revives']]
feature11.fillna(0, inplace=True)

# Moving distance in match max moving distance
df1 = pd.merge(df_all[['Id', 'matchId']], feature8, on='Id')
df2 = df1.groupby(['matchId'], as_index=False).agg({'total_distance':'max'})
df2.rename(columns={'total_distance':'max_total_distance'}, inplace=True)
df3 = pd.merge(df1, df2, on='matchId', how='left')
df3.loc[:, 'distance_in_max_distance'] = df3.total_distance / df3.max_total_distance
feature12 = df3[['Id', 'distance_in_max_distance']]
feature12.fillna(0, inplace=True)

# Heals in match max heals
df1 = df_all[['Id', 'matchId', 'heals']]
df2 = df1.groupby(['matchId'], as_index=False).agg({'heals':'max'})
df2.rename(columns={'heals':'max_heals'}, inplace=True)
df3 = pd.merge(df1, df2, on='matchId', how='left')
df3.loc[:, 'heals_in_max_heals'] = df3.heals / df3.max_heals
feature13 = df3[['Id', 'heals_in_max_heals']]
feature13.fillna(0, inplace=True)

# Merge personal features
features = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13]

person_features = pd.concat(
    (iDF.set_index('Id') for iDF in features),
    axis=1, join='inner'
).reset_index()

# # Merge person features to original data
df = pd.merge(df_all, person_features, on='Id')
test_id = df[-len(test):].Id.values
df.drop(['Id', 'groupId', 'matchId'], 1, inplace=True)

# One-hot encode for character feature
dummy_features = pd.get_dummies(df['matchType'], prefix='matchType')
for dummy in dummy_features:
    df[dummy] = dummy_features[dummy]
df.drop(['matchType'], 1, inplace=True)

# Normalize numberic value
for index, value in df.dtypes.iteritems():
    if index != 'winPlacePerc':
        minValue = df[index].min()
        maxValue = df[index].max()
        df[index] = (df[index] - minValue) / (maxValue - minValue)
        
# Split train/test
train = df[:-len(test)]
test = df[-len(test):] 

# Modeling:xgboost
yTrain = np.array(train.winPlacePerc)
train.drop(['winPlacePerc'], 1, inplace=True)
test.drop(['winPlacePerc'], 1, inplace=True)

model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=200, max_depth=4, min_child_weight=5, seed=0,
                             subsample=0.7, colsample_bytree=0.7, gamma=0.1, reg_alpha=1, reg_lambda=1)

model.fit(train, yTrain)

yhat = model.predict(test)

# Save result
arr = np.vstack((test_id, yhat))
arr = arr.T
df_final = pd.DataFrame(arr, columns=['Id', 'winPlacePerc'])
df_final.to_csv("submission1121.csv", index=False)
print("Generated submission file!")
        
