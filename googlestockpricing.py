# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:00:13 2019

@author: 91947
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, linear_model

df = pd.read_csv('googlestockpricing.csv', parse_dates = ['Date'])
df.set_index('Date', inplace=True)

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['Adj. Close'] = df['Adj. Close'].rolling(25).mean()

df['Label'] = df['Adj. Close'].shift(-1)

df.dropna(inplace=True)

X = np.array(df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']])
X = preprocessing.scale(X)

y = np.array(df['Label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
accuracy = reg.score(X_test, y_test)





