# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 16:36:34 2022

@author: Prothit Sen and Dinesh Kumar (Indian School of Business)

Project title: Do Alliance Portfolios Encourage or Impede New Business Practice Adoption? 
Theory and Evidence from the Private Equity Industry
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso

# importing the dataset
df = pd.read_excel('InductionSample.xls')
df1 = df.drop(['Firm_dealID'],axis = 1)


Y = df1['v0']

X = df1.iloc[: , 1 :]


v2 = df['v2']
v3 = df['v3']
v4 = df['v4']
v6 = df['v6']
v9 = df['v9']
v17 = df['v17']
v24 = df['v24']
v25 = df['v25']
v28 = df['v28']
v29 = df['v29']




v2v3 = v2*v3
v2v4 = v2*v4
v2v6 = v2*v6
v2v9 = v2*v9
v2v17 = v2*v17
v2v24 = v2*v24
v2v25 = v2*v25
v2v28 = v2*v28
v2v29 = v2*v29
v3v4 = v3*v4
v3v6 = v3*v6
v3v9 = v3*v9
v3v17 = v3*v17
v3v24 = v3*v24
v3v25 = v3*v25
v3v28 = v3*v28
v3v29 = v3*v29
v4v6 = v4*v6
v4v9 = v4*v9
v4v17 = v4*v17
v4v24 = v4*v24
v4v25 = v4*v25
v4v28 = v4*v28
v4v29 = v4*v29
v6v9 = v6*v9
v6v17 = v6*v17
v6v24 = v6*v24
v6v25 = v6*v25
v6v28 = v6*v28
v6v29 = v6*v29
v9v17 = v9*v17
v9v24 = v9*v24
v9v25 = v9*v25
v9v28 = v9*v28
v9v29 = v9*v29
v17v24 = v17*v24
v17v25 = v17*v25
v17v28 = v17*v28
v17v29 = v17*v29
v24v25 = v24*v25
v24v28 = v24*v28
v24v29 = v24*v29
v25v28 = v25*v28
v25v29 = v25*v29
v28v29 = v28*v29

s1 = pd.Series(v2v3 , name = 'v2v3')
s2 = pd.Series(v2v4 , name = 'v2v4')
s3 = pd.Series(v2v6 , name = 'v2v6')
s4 = pd.Series(v2v9 , name = 'v2v9')
s5 = pd.Series(v2v17 , name = 'v2v17')
s6 = pd.Series(v2v24 , name = 'v2v24')
s7 = pd.Series(v2v25 , name = 'v2v25')
s8 = pd.Series(v2v28 , name = 'v2v28')
s9 = pd.Series(v2v29 , name = 'v2v29')
s10 = pd.Series(v3v4 , name = 'v3v4')
s11 = pd.Series(v3v6 , name = 'v3v6')
s12 = pd.Series(v3v9 , name = 'v3v9')
s13 = pd.Series(v3v17 , name = 'v3v17')
s14 = pd.Series(v3v24 , name = 'v3v24')
s15 = pd.Series(v3v25 , name = 'v3v25')
s16 = pd.Series(v3v28 , name = 'v3v28')
s17 = pd.Series(v3v29 , name = 'v3v29')
s18 = pd.Series(v4v6 , name = 'v4v6')
s19 = pd.Series(v4v9 , name = 'v4v9')
s20 = pd.Series(v4v17 , name = 'v4v17')
s21 = pd.Series(v4v24 , name = 'v4v24')
s22 = pd.Series(v4v25 , name = 'v4v25')
s23 = pd.Series(v4v28 , name = 'v4v28')
s24 = pd.Series(v4v29 , name = 'v4v29')
s25 = pd.Series(v6v9 , name = 'v6v9')
s26 = pd.Series(v6v17 , name = 'v6v17')
s27 = pd.Series(v6v24 , name = 'v6v24')
s28 = pd.Series(v6v25 , name = 'v6v25')
s29 = pd.Series(v6v28 , name = 'v6v28')
s30 = pd.Series(v6v29 , name = 'v6v29')
s31 = pd.Series(v9v17 , name = 'v9v17')
s32 = pd.Series(v9v24 , name = 'v9v24')
s33 = pd.Series(v9v25 , name = 'v9v25')
s34 = pd.Series(v9v28 , name = 'v9v28')
s35 = pd.Series(v9v29 , name = 'v9v29')
s36 = pd.Series(v17v24 , name = 'v17v24')
s37 = pd.Series(v17v25 , name = 'v17v25')
s38 = pd.Series(v17v28 , name = 'v17v28')
s39 = pd.Series(v17v29 , name = 'v17v29')
s40 = pd.Series(v24v25 , name = 'v24v25')
s41 = pd.Series(v24v28 , name = 'v24v28')
s42 = pd.Series(v24v29 , name = 'v24v29')
s43 = pd.Series(v25v28 , name = 'v25v28')
s44 = pd.Series(v25v29 , name = 'v25v29')
s45 = pd.Series(v28v29 , name = 'v28v29')

s46 = pd.Series(v2 , name = 'v2')
s47 = pd.Series(v3 , name = 'v3')
s48 = pd.Series(v4 , name = 'v4')
s49 = pd.Series(v6 , name = 'v6')
s50 = pd.Series(v9 , name = 'v9')
s51 = pd.Series(v17 , name = 'v17')
s52 = pd.Series(v24 , name = 'v24')
s53 = pd.Series(v25 , name = 'v25')
s54 = pd.Series(v28 , name = 'v28')
s55 = pd.Series(v29 , name = 'v29')

X = pd.concat([s1,s2,s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,s14, s15, s16, s17, s18, s19
                      ,s20,s21,s22,s23,s24,s25,s26,s27,s28,s29,s30,s31,s32,s33,s34,s35,s36,s37
                      ,s38,s39,s40,s41,s42,s43,s44,s45,s46,s47],s48,s49,s50,s51,s52,s53,s54,s55, axis =1)



colname = list(X.columns.values)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.50, random_state = 0)


lasso = Lasso(alpha=0.0003)
model = lasso.fit(X_train,Y_train)


score2 = model.coef_
importance = pd.Series(score2, colname)
importance.nlargest(35).plot(kind='barh')

train_score = lasso.score(X_train, Y_train)
test_score = lasso.score(X_test, Y_test)

print("training score: ", train_score)
print("test score: ", test_score)
