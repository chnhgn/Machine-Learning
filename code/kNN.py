# -*- coding: utf-8 -*-
import pandas as pd
import os, sys
import numpy as np
from matplotlib import pyplot as plt


class kNN(object):
    
    def normal_data(self, file_name):
        ds_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        file_dir = os.path.join(ds_path, file_name)
        
        # Only need columns playMin, playHeigh, playWeight and playFG
        pre_data = pd.read_csv(file_dir, usecols=[19,20,21,31,32])
        labels = np.array(pd.read_csv(file_dir, usecols=[20]))
        
        arr = np.array(pre_data)
        # The max and min values for each column
        max0 = arr[:,0].max()
        min0 = arr[:,0].min()
    
        max1 = arr[:,1].max()
        min1 = arr[:,1].min()
        
        max2 = arr[:,2].max()
        min2 = arr[:,2].min()
        
        max3 = arr[:,3].max()
        min3 = arr[:,3].min()
        
        # Normalize each element
        for row in arr:
            row[0] = (row[0] - min0)/(max0 - min0)
            row[1] = (row[1] - min1)/(max1 - min1)
            row[2] = (row[2] - min2)/(max2 - min2)
            row[3] = (row[3] - min3)/(max3 - min3)
      
        return arr

    def trainTestSplit(self, X,test_size=0.3):
        X_num=X.shape[0]
        train_index=list(range(X_num))
        test_index=[]
        test_num=int(X_num*test_size)
        
        for i in range(test_num):
            randomIndex=int(np.random.uniform(0,len(train_index)))
            test_index.append(train_index[randomIndex])
            del train_index[randomIndex]
        X = pd.DataFrame(X)
        train=X.ix[train_index] 
        test=X.ix[test_index]
        return np.array(train),np.array(test)

    def classify_test(self, file_name):
        data = self.normal_data(file_name)
        train, test = self.trainTestSplit(data, 0.1)
        train_rows = train.shape[0]
        cmp_list = []
        
        for row in test:
            a = np.tile(row[:-1], (train_rows, 1))
            temp = train
            b = np.delete(temp, -1, axis=1)
            c = a - b
            distance = np.sum(c**2, axis=1)**0.5
            labels = train.T[-1]
            li = []
            for index, item in enumerate(labels):
                li.append((distance[index], item))
            # Find the top k
            li.sort(key=lambda x:x[0])    
            cmp_list.append((li[0][1], row[-1]))
        
        return cmp_list
    
    def accurate_rate(self, in_list):
        total = len(in_list)
        count = 0
        for item in in_list:
            if item[0] == item[1]:
                count += 1
    
        return str("%.4f" % (count/total))
    
    def plot(self, file_name):
        ds_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        file_dir = os.path.join(ds_path, file_name)
        
        # Only need columns playMin, playHeigh, playWeight and playFG
        pre_data = pd.read_csv(file_dir, usecols=[20,21,32])
        arr = np.array(pre_data)
             
        plt.figure(figsize=(8, 5), dpi=80)
        axes = plt.subplot(111)
         
        type1_x = []
        type1_y = []
        type2_x = []
        type2_y = []
        type3_x = []
        type3_y = []
        type4_x = []
        type4_y = []
        type5_x = []
        type5_y = []
         
        for item in arr:
            if item[2] == 'PF':
                type1_x.append(item[0])
                type1_y.append(item[1])
            elif item[2] == 'SF':
                type2_x.append(item[0])
                type2_y.append(item[1])  
            elif item[2] == 'PG':
                type3_x.append(item[0])
                type3_y.append(item[1])
            elif item[2] == 'SG':
                type4_x.append(item[0])
                type4_y.append(item[1]) 
            elif item[2] == 'C':
                type5_x.append(item[0])
                type5_y.append(item[1])           
         
        type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
        type2 = axes.scatter(type2_x, type2_y, s=30, c='green')
        type3 = axes.scatter(type3_x, type3_y, s=40, c='blue')
        type4 = axes.scatter(type4_x, type4_y, s=50, c='yellow')
        type5 = axes.scatter(type5_x, type5_y, s=60, c='black')
         
        plt.xlabel('Height')
        plt.ylabel('Weight')
        axes.legend((type1, type2, type3, type4, type5), ('PF', 'SF', 'PG', 'SG', 'C'), loc=2)
         
        plt.show()
    
    
    
    
    
if __name__ == '__main__':
    
    knn = kNN()   
    # Plot the trend for training data
    knn.plot("2016-17_playerBoxScore.csv") 
    
    res = knn.classify_test("2016-17_playerBoxScore.csv")
    rate = knn.accurate_rate(res)
    print("The accurate rate is %s" % rate)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    