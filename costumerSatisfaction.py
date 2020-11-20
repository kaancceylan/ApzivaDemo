# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 12:42:32 2020

@author: Kaan
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

#I will be using the decisiontreeclassifier since this will be
#a binary classification and we have a considerably small amount of data 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


costumer_satisfaction_data = pd.read_csv('ACME-HappinessSurvey2020.csv')

#Independent Variables
X = costumer_satisfaction_data.iloc[:, 1:]

#Dependent Variable
Y = costumer_satisfaction_data.iloc[:, 0:1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)


#Implementing the algorithm
#The default of 100 n_estimators value is too high for this data so we will adjust that
#And we will leave the splitting criterion as gini

rfc = RandomForestClassifier(n_estimators = 18, max_depth=5)
rfc.fit(X_train, Y_train)
Y_pred = rfc.predict(X_test)


#Accuracy of rfc
acc_score = round(rfc.score(X_train, Y_train)*100, 2)
print(f'Random Forest Accuracy: {acc_score}')


#Running feature importances
feature_imp = pd.DataFrame({'Feature':X_train.columns, 
                            'Importance Score':np.round(rfc.feature_importances_, 3)})

feature_imp = feature_imp.sort_values('Importance Score', ascending = False).set_index('Feature')

print(feature_imp)


#We can drop the feature X6
X = X.drop('X6', axis=1)

#Cross validate
cv_score = cross_val_score(rfc, X_train, Y_train, cv=4, scoring='accuracy')
print('Cross Validation Score:', cv_score)
print('Mean:', cv_score.mean())
print('Standart Deviation:', cv_score.std())


#Fit and predict again
rand_forest = RandomForestClassifier(n_estimators=18, max_depth=5)
rand_forest.fit(X_train,Y_train)
Y_predict = rand_forest.predict(X_test)

acc_score = round(rand_forest.score(X_train, Y_train)*100, 2)
print(f'Random Forest Accuracy: {acc_score}')
