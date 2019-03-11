# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:47:43 2019

@author: Ismail
"""

import pandas as pd
import numpy as np

#Read file
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#concatinate train and test
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train,test], ignore_index  = True)
#print(train.shape, test.shape, data.shape)

#print(data.apply(lambda x: sum(x.isnull())))

#basic stat and numerical variable

#print(data.describe())

#number of unique variable

#print(data.apply(lambda x: len(x.unique())))

#filter categorical variable
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x] == 'object']

#Exclude ID col and source
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier', 'Outlet_Identifier', 'source']]

#Print frequency of categories
for col in categorical_columns:
    print('\nFrequency of Categories for variable %s'%col)
    print(data[col].value_counts())
    

#3.CLEANSING
#Determine the average weight per item

item_avg_weight = data.pivot_table(values = 'Item_Weight', index = 'Item_Identifier')

#Get a boolean variable for missing Item_Weight value

miss_bool = data['Item_Weight'].isnull()

#Imoute data and check missing values before and after imputation to confirm
print()
print('Original #missing: %d'%sum(miss_bool))

#data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight[x])
data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.at[x,'Item_Weight'])

print('Final #missing: %d'% sum(data['Item_Weight'].isnull()))
      
