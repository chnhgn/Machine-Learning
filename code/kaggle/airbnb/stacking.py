# -*- coding: utf-8 -*-
'''
Created on Nov 9, 2018

@author: Eddy Hu
'''
import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold



train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
test_id = test.id

lbl = preprocessing.LabelEncoder() 
lbl.fit(list(train.country_destination.values)) 
train.country_destination = lbl.transform(list(train.country_destination.values))

yTrain = np.array(train.country_destination)
train.drop(['country_destination', 'id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)


def gen_result(test_id, yhat):
    s = pd.Series(yhat, name='label')
    df = pd.DataFrame({'id' : test_id, 'label' : s})
    df.label = df.label.astype('int')
    df['country'] = lbl.inverse_transform(df.label)
    df.drop(['label'], axis=1, inplace=True)
    return df 


'''
    Stacking: two layer
'''
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 2018  # for reproducibility
NFOLDS = 5  # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self, x, y):
        return self.clf.fit(x, y)
    
    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)
    
# Class to extend XGboost classifer

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Random Forest parameters
rf_params = {
    'n_jobs':-1,
    'n_estimators': 20,
     'warm_start': True,
     # 'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs':-1,
    'n_estimators':20,
    # 'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 20,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 20,
     # 'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = yTrain
x_train = train.values  # Creates an array of the train data
x_test = test.values  # Creats an array of the test data

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)  # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)  # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)  # AdaBoost 
# gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)  # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)  # Support Vector Classifier

print("Training is complete")

# x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
# x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, svc_oof_test), axis=1)


gbm = xgb.XGBClassifier(
    # learning_rate = 0.02,
     n_estimators=150,
     max_depth=4,
     min_child_weight=2,
     # gamma=1,
     gamma=0.1,
     subsample=0.7,
     colsample_bytree=0.7,
     objective='multi:softmax',
     nthread=4,
     scale_pos_weight=1).fit(x_train, y_train)
 
predictions = gbm.predict(x_test)

result = gen_result(test_id, predictions)
result.to_csv('./data/submission_stacking.csv', index=False)
print('stacking finished!')


