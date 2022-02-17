# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:06:41 2021

@author: Prothit Sen and Dinesh Kumar (Indian School of Business)

Project title: Do Alliance Portfolios Encourage or Impede New Business Practice Adoption? 
Theory and Evidence from the Private Equity Industry
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
df = pd.read_excel('InductionSample_v2.xls')

X =df[['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v13','v14','v15','v16','v17','v18','v19','v20',
        'v21','v22','v23','v24','v25','v26','v27','v28','v29']] 
Y = df['v0']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.5, random_state = 0)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',random_state = 0 , max_depth = 4)
classifier = classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the confusion library
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

print("Accuracy: ",accuracy_score(y_test, y_pred))



# Plotting variable importance plot

model = classifier.fit(X_train, y_train)
feature_imp = model.feature_importances_
feat_importances = pd.Series(feature_imp, X_train.columns.tolist())
feat_importances.nlargest(29).plot(kind='barh')
