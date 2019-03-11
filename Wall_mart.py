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

#Impute outlet_size

from scipy.stats import mode


#Determinig the mode for each
#outlet_size_mode = data.pivot_table(values = 'Outlet_Size', columns = 'Outlet_Type', aggfunc = (lambda x:mode(x).mode[0]))
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:x.mode().iat[0]) )
print('Mode for each Outlet_Type:')
print(outlet_size_mode)  


#get the boolean variable

miss_bool = data['Outlet_Size'].isnull()

#Impute data and check missing values
print('\noriginal #missing: %d'%sum(miss_bool))
data.loc[miss_bool, 'Outlet_Size'] = data.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
print(sum(data['Outlet_Size'].isnull()))


"""
4.FEATURE ENGINEERING
making the data ready for analysis,Create some new variable using the old ones"""

#Step 1: Consider combining outlet type

data.pivot_table(values = 'Item_Outlet_Sales',index = 'Outlet_Type')
