#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 18:06:02 2019

@author: venkat
"""


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv("https://raw.githubusercontent.com/trainervenkat/MDUSIT/master/datasets/credit-data.csv")


features = data[["income","age","loan"]]
target = data.default

feature_train, feature_test,target_train, target_test = train_test_split(features,target, test_size=0.3)

#gaussian naive bayes ---gaussian means that features are normal distributed
model = GaussianNB()
fittedModel = model.fit(feature_train, target_train)
predictions = fittedModel.predict(feature_test)

print('confusion matrix',confusion_matrix(target_test, predictions))
print('accuracy score',accuracy_score(target_test,predictions))