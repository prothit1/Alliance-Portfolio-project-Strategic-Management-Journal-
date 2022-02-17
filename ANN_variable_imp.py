# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 15:17:00 2022

@author: Prothit Sen and Dinesh Kumar (Indian School of Business)

Project title: Do Alliance Portfolios Encourage or Impede New Business Practice Adoption? 
Theory and Evidence from the Private Equity Industry
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense , Dropout
from keras.wrappers.scikit_learn import KerasClassifier , KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance

# Importing the dataset
df = pd.read_excel('InductionSample_v2.xls')
df1 = df.drop(['Firm_dealID'],axis = 1)

X =df1[['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v13','v14','v15','v16','v17','v18','v19','v20',
        'v21','v22','v23','v24','v25','v26','v27','v28','v29']] 
#

Y = df1['v0']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.5, random_state = 0)

# Defining variables
input = X_train.shape[1]
epochs = 50
learning_rate = 0.1
decay_rate = learning_rate / epochs
batch_size = 15

 
# Initialising the ANN
def my_model(units):
    classifier = Sequential()

# Adding the input layer and the first hidden layer
    classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = 'relu', input_dim = 33))

# Adding the second hidden layer
 #   classifier.add(Dropout(rate = dropout_rate))
    classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier


# Fitting the ANN to the Training set


# model to run
classifier = KerasClassifier(build_fn = my_model)


from sklearn.model_selection import GridSearchCV

batch_size = 10 
epochs = 50
units = 6

param_grid = dict(batch_size=batch_size, epochs=epochs,units = units)
gridSearch = GridSearchCV(estimator=classifier,
                          param_grid=param_grid,
                          cv=2,
                         n_jobs=1,
                         return_train_score=True)

gridSearch.fit(X_train,y_train)

print('Grid Search Best score',gridSearch.best_score_)
print('Grid Search Best Parameters', gridSearch.best_params_)




df_gridsearch = pd.DataFrame(gridSearch.cv_results_)

from mpl_toolkits import mplot3d
ax = plt.axes(projection = "3d")

# Data for a three-dimensional line
zline = df_gridsearch['param_batch_size']
xline = df_gridsearch[['param_epochs'] == 50].min()
yline = df_gridsearch['mean_train_score']

ax.plot3D(xline, yline, zline, 'gray')
plt.show()




df_gridsearch.to_excel("result.xls")

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



# Permutation Importance
perm = PermutationImportance(classifier, random_state=1).fit(X_train, y_train)
eli5.show_weights(perm, show_feature_values=True,feature_names = X_train.columns.tolist())


