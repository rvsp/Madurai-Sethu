#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 22:14:53 2019

@author: venkat
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 19:03:23 2019

@author: venkat
"""

# Multiple Linear Regression

# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math

hire = pd.read_csv("https://raw.githubusercontent.com/trainervenkat/MDUSIT/master/datasets/hiring.csv")
print(hire)



median_test_score = math.floor(hire['test_score(out of 10)'].mean())
hire['test_score(out of 10)'] = hire['test_score(out of 10)'].fillna(median_test_score)
print(median_test_score)


reg = LinearRegression()

reg.fit(hire[['experience','test_score(out of 10)','interview_score(out of 10)']],hire['salary($)'])
reg.coef_
reg.intercept_

to_predict = [
        [2,9,6],
        [7,9,10]
        ]

reg.predict(to_predict)