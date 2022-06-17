# CKD Prediction by using KNN Algorithm
#libraries
import pandas as pd
import numpy as np
from numpy import nan
import matplotlib.pyplot as plt

#preprocessing and split libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#confusion && accuracy &&classification_report libraries
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# import dataset
missing_values = ["n/a", "na", "?", "missing"]
ds = pd.read_csv("chronic kidney disease.csv", na_values = missing_values)


print("the shape of dataset = ",ds.shape)


print(ds.info())

print("Print the dataset \n",ds.head(20))

print("Before filling in the missing values \n",ds.isnull().sum())

ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]]= ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]].replace('yes', 1)
ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]]= ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]].replace('no', 0)

ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]]= ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]].replace('abnormal', 1)
ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]]= ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]].replace('normal', 0)

ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]]= ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]].replace('present', 1)
ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]]= ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]].replace('notpresent', 0)

ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]]= ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]].replace('poor', 1)
ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]]= ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]].replace('good', 0)

ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]]= ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]].replace('ckd', 1)
ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]]= ds[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', "class"]].replace('notckd', 0)


# To fill in the missing values
age = ds["age"].mean()
ds["age"].fillna(age, inplace = True)

bp = ds["bp"].median()
ds["bp"].fillna(bp, inplace = True)

sg = ds["sg"].mean()
ds["sg"].fillna(sg, inplace = True)

al = ds["al"].median()
ds["al"].fillna(al, inplace = True)

su = ds["su"].median()
ds["su"].fillna(su, inplace = True)

rbc = ds["rbc"].mode()[0]
ds["rbc"].fillna(rbc, inplace = True)

pc = ds["pc"].mode()[0]
ds["pc"].fillna(pc, inplace = True)

pcc = ds["pcc"].mode()[0]
ds["pcc"].fillna(pcc, inplace = True)

ba = ds["ba"].mode()[0]
ds["ba"].fillna(ba, inplace = True)

bgr = ds["bgr"].median()
ds["bgr"].fillna(bgr, inplace = True)

bu = ds["bu"].median()
ds["bu"].fillna(bu, inplace = True)

sc = ds["sc"].mean()
ds["sc"].fillna(sc, inplace = True)

sod = ds["sod"].median()
ds["sod"].fillna(sod, inplace = True)

pot = ds["pot"].mean()
ds["pot"].fillna(pot, inplace = True)

hemo = ds["hemo"].mean()
ds["hemo"].fillna(bu, inplace = True)

pcv = ds["pcv"].median()
ds["pcv"].fillna(pcv, inplace = True)

wbcc = ds["wbcc"].mean()
ds["wbcc"].fillna(wbcc, inplace = True)

rbcc = ds["rbcc"].mean()
ds["rbcc"].fillna(rbcc, inplace = True)

htn = ds["htn"].mode()[0]
ds["htn"].fillna(htn, inplace = True)

dm = ds["dm"].mode()[0]
ds["dm"].fillna(dm, inplace = True)

cad = ds["cad"].mode()[0]
ds["cad"].fillna(cad, inplace = True)

appet = ds["appet"].mode()[0]
ds["appet"].fillna(appet, inplace = True)

pe = ds["pe"].mode()[0]
ds["pe"].fillna(pe, inplace = True)

ane = ds["ane"].mode()[0]
ds["ane"].fillna(ane, inplace = True)

print("After filling in the missing values \n",ds.isnull().sum())


#split dataset to features and target
target_name ='class'
data_target =ds[target_name]
data =ds.drop([target_name], axis=1)

# Data Normalization
x_scaler = MinMaxScaler()
x_scaler.fit(data)
column_names = data.columns
data[column_names] = x_scaler.transform(data)

#split dataset set to training (80%) and testing (20%)
train, test, target,target_test =train_test_split(data, data_target, test_size=0.2, random_state=0)

print("Training dataset after preprocessing data \n",train.head(20))

# KNeighborsClassifier
print("KNeighborsClassifier")
knn = KNeighborsClassifier(n_neighbors=6,weights='distance',algorithm='auto',p=2)
# fit the model on training data
knn.fit(train,target)

prediction=knn.predict(test)
#print("prediction =\n",prediction)

# score of prediction
print("accuracy_of_KNN =", accuracy_score(target_test, prediction)*100)
print("confusion_matrix_of_KNN = \n", confusion_matrix(target_test, prediction))
print("classification_report_of_KNN =\n", classification_report(target_test, prediction))
