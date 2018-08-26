# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sb

##Set directory
path="C:/F/NMIMS/DataScience/Sem-3/Projects/BigMartSales/data"
os.chdir(path)

##Read CSV
bigMart_data = pd.read_csv("Train.csv")
bigMart_data.info()
bigMart_data.describe()
bigMart_data.head(3)
bigMart_data.tail(3)
bigMart_data.isnull().sum()

##Cleaning data
def clean_ItemVisibility(itemVis):
    if itemVis == 0:
        return np.mean(bigMart_data['Item_Visibility'])
        
    else:
        return itemVis
    
bigMart_data['Item_Visibility'] = bigMart_data['Item_Visibility'].apply(clean_ItemVisibility)

def clean_ItemFatContent(fat):
    fatContent=""
    if fat == "LF":
        fatContent = "Low Fat"
    elif fat == "reg":
        fatContent = "Regular"
    else:
        fatContent = fat
    return fatContent.lower()
    
bigMart_data['Item_Fat_Content'] = bigMart_data['Item_Fat_Content'].apply(clean_ItemFatContent)

def clean_ItemWeight(itemWt):
    if pd.isnull(itemWt):
        return np.mean(bigMart_data['Item_Weight'])
        
    else:
        return itemWt
    
bigMart_data['Item_Weight'] = bigMart_data['Item_Weight'].apply(clean_ItemWeight)

bigMart_data['Outlet_Size'].value_counts()
bigMart_data.loc[bigMart_data['Outlet_Size'].isnull(),"Outlet_Identifier"].value_counts()

bigMart_data.loc[(bigMart_data['Outlet_Location_Type'] == 'Tier 3') & (bigMart_data['Outlet_Type'] == 'Grocery Store'), "Outlet_Size"].value_counts()
bigMart_data.loc[(bigMart_data['Outlet_Type'] == 'Grocery Store'), "Outlet_Size"].value_counts()
bigMart_data.loc[(bigMart_data['Outlet_Location_Type'] == 'Tier 3'), "Outlet_Size"].value_counts()


bigMart_data.loc[(bigMart_data["Outlet_Location_Type"]== "Tier 2") & (bigMart_data["Outlet_Type"]=="Supermarket Type1") ,"Outlet_Size"].value_counts()

def clean_OutletSize(cols):
    identifier=cols[0]
    size=cols[1]
    if (identifier == 'OUT017') | (identifier == 'OUT045'):
        return "Small"
        
    elif identifier == 'OUT010':
        return "Small"
    else:
        return size
    
bigMart_data["Outlet_Size"] = bigMart_data[["Outlet_Identifier", "Outlet_Size"]].apply(clean_OutletSize, axis=1)

bigMart_data.Outlet_Location_Type.value_counts()
bigMart_data.Item_Type.value_counts()
bigMart_data.Item_Fat_Content.value_counts()
bigMart_data.Outlet_Type.value_counts()

##Replacing categorical variables with dummies
fat_content = pd.get_dummies(bigMart_data['Item_Fat_Content'], drop_first=True)
fat_content.head()
item_type = pd.get_dummies(bigMart_data['Item_Type'], drop_first=True)
item_type.head()
outlet_size = pd.get_dummies(bigMart_data['Outlet_Size'], drop_first=True)
outlet_size.head()
loc_type = pd.get_dummies(bigMart_data['Outlet_Location_Type'], drop_first=True)
loc_type.head()
outlet_type = pd.get_dummies(bigMart_data['Outlet_Type'], drop_first=True)
outlet_type.head()

bigMart_data.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'], 1, inplace=True)
bigMart_data_dmy = pd.concat([fat_content, item_type, outlet_size, loc_type, outlet_type, bigMart_data], axis=1)
bigMart_data_dmy.info()
bigMart_data_dmy.head(1)

##Plot heat Map
sb.heatmap(bigMart_data.corr())


bigMart_data_dmy.columns

Xcols=['regular', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods',
       'Fruits and Vegetables', 'Hard Drinks', 'Health and Hygiene',
       'Household', 'Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks',
       'Starchy Foods', 'Medium', 'Small', 'Tier 2', 'Tier 3',
       'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3',
       'Item_Weight', 'Item_Visibility', 'Item_MRP',
       'Outlet_Establishment_Year']
Ycols='Item_Outlet_Sales'

X = pd.DataFrame(bigMart_data_dmy.iloc[:,:-1].values, columns=Xcols)
y = bigMart_data_dmy.iloc[:,-1:].values


#Splitting into train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Preprocessing
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score

####Linear Regression with all columns
reg = linear_model.LinearRegression(fit_intercept=True, normalize=True)
reg.fit(X_train, y_train)
print(reg.coef_)
print(reg.intercept_)
y_predicted = reg.predict(X_test)
metrics.mean_squared_error(y_true=y_test, y_pred=y_predicted)
r2_score(y_test, y_predicted)

### Lasso Model
Lasso = linear_model.LassoCV(cv=5,normalize=True,random_state=10,alphas=[.0005])
Lasso
Lasso.fit(X_train, y_train)
print(Lasso.intercept_)
coef1 = pd.DataFrame(Lasso.coef_,Xcols,columns=["Value"])
coef1[coef1["Value"]>0].sort_values(by="Value",ascending=False)

y_pred = Lasso.predict(X_test)
print(metrics.mean_squared_error(y_true=y_test, y_pred=y_pred))
print(r2_score(y_test, y_pred))







### Predicting the Sales for unknown data set ########
##Read CSV
bigMart_testData = pd.read_csv("Test_1.csv")
bigMart_testData.info()
bigMart_testData.describe()
bigMart_testData.head(3)
bigMart_testData.tail(3)
bigMart_testData.isnull().sum()

##Cleaning data
def clean_testItemVisibility(itemVis):
    if itemVis == 0:
        return np.mean(bigMart_testData['Item_Visibility'])
        
    else:
        return itemVis
    
bigMart_testData['Item_Visibility'] = bigMart_testData['Item_Visibility'].apply(clean_testItemVisibility)

def clean_testItemFatContent(fat):
    fatContent=""
    if fat == "LF":
        fatContent = "Low Fat"
    elif fat == "reg":
        fatContent = "Regular"
    else:
        fatContent = fat
    return fatContent.lower()
    
bigMart_testData['Item_Fat_Content'] = bigMart_testData['Item_Fat_Content'].apply(clean_testItemFatContent)

def clean_testItemWeight(itemWt):
    if pd.isnull(itemWt):
        return np.mean(bigMart_testData['Item_Weight'])
        
    else:
        return itemWt
    
bigMart_testData['Item_Weight'] = bigMart_testData['Item_Weight'].apply(clean_testItemWeight)

bigMart_testData['Outlet_Size'].value_counts()
bigMart_testData.loc[bigMart_testData['Outlet_Size'].isnull(),"Outlet_Identifier"].value_counts()

bigMart_testData.loc[(bigMart_testData['Outlet_Location_Type'] == 'Tier 3') & (bigMart_testData['Outlet_Type'] == 'Grocery Store'), "Outlet_Size"].value_counts()
bigMart_testData.loc[(bigMart_testData['Outlet_Type'] == 'Grocery Store'), "Outlet_Size"].value_counts()
bigMart_testData.loc[(bigMart_testData['Outlet_Location_Type'] == 'Tier 3'), "Outlet_Size"].value_counts()


bigMart_testData.loc[(bigMart_testData["Outlet_Location_Type"]== "Tier 2") & (bigMart_testData["Outlet_Type"]=="Supermarket Type1") ,"Outlet_Size"].value_counts()

def clean_testOutletSize(cols):
    identifier=cols[0]
    size=cols[1]
    if (identifier == 'OUT017') | (identifier == 'OUT045'):
        return "Small"
        
    elif identifier == 'OUT010':
        return "Small"
    else:
        return size
    
bigMart_testData["Outlet_Size"] = bigMart_testData[["Outlet_Identifier", "Outlet_Size"]].apply(clean_testOutletSize, axis=1)

bigMart_testData.Outlet_Location_Type.value_counts()
bigMart_testData.Item_Type.value_counts()
bigMart_testData.Item_Fat_Content.value_counts()
bigMart_testData.Outlet_Type.value_counts()

##Replacing categorical variables with dummies
fat_testContent = pd.get_dummies(bigMart_testData['Item_Fat_Content'], drop_first=True)
fat_testContent.head()
item_testType = pd.get_dummies(bigMart_testData['Item_Type'], drop_first=True)
item_testType.head()
outlet_testSize = pd.get_dummies(bigMart_testData['Outlet_Size'], drop_first=True)
outlet_testSize.head()
loc_testType = pd.get_dummies(bigMart_testData['Outlet_Location_Type'], drop_first=True)
loc_testType.head()
outlet_testType = pd.get_dummies(bigMart_testData['Outlet_Type'], drop_first=True)
outlet_testType.head()

bigMart_testData.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'], 1, inplace=True)
bigMart_testData_dmy = pd.concat([fat_testContent, item_testType, outlet_testSize, loc_testType, outlet_testType, bigMart_testData], axis=1)
bigMart_testData_dmy.info()
bigMart_testData_dmy.head(1)


y_testpred = Lasso.predict(bigMart_testData_dmy)
y_testpred = pd.DataFrame(y_testpred)
y_testpred.info()


csv_input = pd.read_csv('Test_1.csv')
csv_input['Item_Outlet_Sales'] = y_testpred
csv_input.to_csv('Test.csv', index=False)






