#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 16:04:10 2018

@author: Shashank Pawar
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Categorical Data Encoding
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])

# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder_X = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder_X.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap 
"""Not required explicitly
Python libraries ususally take care of it
"""
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""Not required explicitly
Python libraries ususally take care of it
"""

# Training the model
from sklearn.linear_model import LinearRegression
MLR = LinearRegression()
MLR.fit(X_train, y_train)

# Prediciting on the test set
y_pred = MLR.predict(X_test)


# Using Backward Elimination for building Optimal MLR
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values=X,axis=1)
X_opt = X[:,[0,1,2,3,4,5]]
MLR_OLS = sm.OLS(endog=y, exog=X_opt).fit()
MLR_OLS.summary()
"""
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.013e+04   6884.820      7.281      0.000    3.62e+04     6.4e+04
x1           198.7888   3371.007      0.059      0.953   -6595.030    6992.607
x2           -41.8870   3256.039     -0.013      0.990   -6604.003    6520.229
x3             0.8060      0.046     17.369      0.000       0.712       0.900
x4            -0.0270      0.052     -0.517      0.608      -0.132       0.078
x5             0.0270      0.017      1.574      0.123      -0.008       0.062
==============================================================================
"""
# x2 has the largest P-Value. Hence removing it and refitting the model
X_opt = X[:,[0,1,3,4,5]]
MLR_OLS = sm.OLS(endog=y, exog=X_opt).fit()
MLR_OLS.summary()
"""
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.011e+04   6647.870      7.537      0.000    3.67e+04    6.35e+04
x1           220.1585   2900.536      0.076      0.940   -5621.821    6062.138
x2             0.8060      0.046     17.606      0.000       0.714       0.898
x3            -0.0270      0.052     -0.523      0.604      -0.131       0.077
x4             0.0270      0.017      1.592      0.118      -0.007       0.061
==============================================================================
"""

# x1 has the largest P-Value. Hence removing it and refitting the model
X_opt = X[:,[0,3,4,5]]
MLR_OLS = sm.OLS(endog=y, exog=X_opt).fit()
MLR_OLS.summary()
"""
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.012e+04   6572.353      7.626      0.000    3.69e+04    6.34e+04
x1             0.8057      0.045     17.846      0.000       0.715       0.897
x2            -0.0268      0.051     -0.526      0.602      -0.130       0.076
x3             0.0272      0.016      1.655      0.105      -0.006       0.060
==============================================================================
"""
# x2 has the largest P-Value. Hence removing it and refitting the model
X_opt = X[:,[0,3,5]]
MLR_OLS = sm.OLS(endog=y, exog=X_opt).fit()
MLR_OLS.summary()
"""
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.698e+04   2689.933     17.464      0.000    4.16e+04    5.24e+04
x1             0.7966      0.041     19.266      0.000       0.713       0.880
x2             0.0299      0.016      1.927      0.060      -0.001       0.061
==============================================================================
"""

# x2 has the largest P-Value. Hence removing it and refitting the model
X_opt = X[:,[0,3]]
MLR_OLS = sm.OLS(endog=y, exog=X_opt).fit()
MLR_OLS.summary()
"""
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.903e+04   2537.897     19.320      0.000    4.39e+04    5.41e+04
x1             0.8543      0.029     29.151      0.000       0.795       0.913
"""

# Now no independent variable has P-Value > SL. Hence our model is ready
p_vals = MLR_OLS.pvalues







