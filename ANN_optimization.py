# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 15:14:59 2022

@author: Prothit Sen and Dinesh Kumar (Indian School of Business)

Project title: Do Alliance Portfolios Encourage or Impede New Business Practice Adoption? 
Theory and Evidence from the Private Equity Industry
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier , KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance


# Importing the dataset
df = pd.read_excel('InductionSample.xls')
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
batch_size = 15


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = input))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the fourth hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


lr_model = classifier.fit(X_train, y_train, batch_size = batch_size, epochs = epochs , validation_data = (X_test,y_test))



# Plot the loss function
fig, ax = plt.subplots(2, 1, figsize=(10,6),constrained_layout=True)
ax[0].plot(np.sqrt(lr_model.history['loss']), 'r', label='train')
ax[0].plot(np.sqrt(lr_model.history['val_loss']), 'b' ,label='val')
ax[0].set_xlabel(r'Epoch', fontsize=16)
ax[0].set_ylabel(r'Loss', fontsize=16)
ax[0].legend()
ax[0].tick_params(labelsize=20)
ax[0].set_title("No. of Epochs vs Loss ")

# Plot the accuracy
#fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax[1].plot(np.sqrt(lr_model.history['accuracy']), 'r', label='train')
ax[1].plot(np.sqrt(lr_model.history['val_accuracy']), 'b' ,label='val')
ax[1].set_xlabel(r'Epoch', fontsize=16)
ax[1].set_ylabel(r'Accuracy', fontsize=16)
ax[1].legend()
ax[1].tick_params(labelsize=20)
ax[1].set_title("No. of Epoch vs Accuracy")