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


#Step 2: modify th e item visibility
#here minimum value is 0 so impute it with mean visibility of that product

#average visibility of a product
visibility_avg = data.pivot_table(values = 'Item_Visibility', index = 'Item_Identifier')

#impute 0 values with mean visibility of the product
miss_bool = (data['Item_Visibility'] == 0)

#print number of 0
print('\nNumber of 0 values initially: %d'%sum(miss_bool))

#impute mean visibility
data.loc[miss_bool,'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.at[x,'Item_Visibility'])
print('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))



#Determine another variable with means ratio
data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg.loc[x['Item_Identifier']], axis=1)

print(data['Item_Visibility_MeanRatio'].describe())


#Step 3: CREATE A BROAD CATEGORY OF ITEM

#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])

#Rename them to more intuitive categories
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD': 'Food',
    'NC': 'Non-Consumable',
    'DR' : 'Drinks'})
    
print()
print(data['Item_Type_Combined'].value_counts())


#Step 4: DETERMINE THE YEAR OF OPERATION OF A STORE

"""We wanted to make a new column depicting the years of operation of a store"""

#years
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']

print()
print(data['Outlet_Years'].describe())

#Step 5: MODIFY CATEGORIES OF ITEM_FAT_CONTENT
"""We found typos and difference in representation in categories 
of Item_Fat_Content variable. 
This can be corrected as"""

#change categories of low fat:

print()
print('Original categories: ')
print(data['Item_Fat_Content'].value_counts())

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat',
    'reg': 'Regular',
    'low fat': 'Low Fat'})

print()
print('Modified categories: ')
print(data['Item_Fat_Content'].value_counts())

#make non-consumable low fat as a separate category in low_fat
data.loc[data['Item_Type_Combined'] == "Non-Consumable",'Item_Fat_Content'] = "Non_Edible"
print()
print(data['Item_Fat_Content'].value_counts())


#Step 6: NUMERICAL AND ONE-HOT CODING OF CATEGORICAL VARIABLES
"""Since scikit-learn accepts only numerical variables,
 I converted all categories of nominal variables into 
 numeric types. Also, I wanted Outlet_Identifier as a
 variable as well. So I created a new variable ‘Outlet’
 same as Outlet_Identifier and coded that. Outlet_Identifier
 should remain as it is, 
because it will be required in the submission file."""

#Importe library
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

#new variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])

var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])
