#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 19:03:23 2019

@author: venkat
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

home= pd.read_csv('https://raw.githubusercontent.com/trainervenkat/MDUSIT/master/datasets/homeprice.csv')
print(home)

home.bedrooms.median()
home.bedrooms = home.bedrooms.fillna(home.bedrooms.median())

reg = LinearRegression()
reg.fit(home.drop('price',axis='columns'),home.price)


reg.coef_
reg.intercept_

to_predict = [
        [3000, 3, 40],
        [2500, 4, 5]
        ]

# this result predicted value
reg.predict(to_predict)


# 498408.25157402386
# 112.06244194*3000 + 23388.88007794*3 + -3231.71790863*40 + 221323.00186540384
