#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 16:07:58 2022

@author: ruslebiffen
"""

import numpy             as np
import pandas            as pd
import seaborn           as sns
import matplotlib.pyplot as plt
import statsmodels.api   as sm
from scipy.optimize      import minimize


x = np.linspace(-10 ,10 ,500)

def objective(x):
    return -(x**2 - 3*x + 2)

y = objective(x)

plt.plot(x,y)


x0 = [0]



def objective(x):
    '''
    Problem 3
    '''
    return (x**2)/(x**2 + 2)

y = objective(x)
plt.plot(x,y)

minimize(objective,x0,method='Nelder-Mead')




N1 = 100
N2 = 250
N3 = 1000
N4 = 10000
N5 = 100000

N = N1


alpha = 0.2
beta  = 0.5

epsilon = np.random.randn(N)  

x = np.random.randn(N)

y = alpha + x*beta + epsilon

X = sm.add_constant(x)


# OLS
XX = np.matmul(X.T,X)

beta_hat = np.linalg.inv(XX)@(X.T@y)
var_beta = np.var(epsilon)*np.linalg.inv(XX)


# Sampling properties of beta_hat
b = [0]*1000
a = [0]*1000

for ii in range(1000):
    epsilon = np.random.randn(N)  
    
    x  = np.random.randn(N)
    
    y  = alpha + x*beta + epsilon*3
    
    X  = sm.add_constant(x)
    
    XX = X.T@X
    
    a[ii] = (np.linalg.inv(XX)@(X.T@y))[0]    
    b[ii] = (np.linalg.inv(XX)@(X.T@y))[1]
ab           = a+b
t1           = ['a']*1000
t2           = ['b']*1000
t            = t1+t2
data         = pd.DataFrame({'Estimate':ab})
data['type'] = t

# Plot histogram
plt.figure()
ax=sns.histplot(data[data.type=='b'].Estimate,kde=True,color='pink',alpha=0.8)
ax.set(title   = 'N = '+str(N))
plt.axvline(data[data.type=='b'].Estimate.mean(), color='black')
#ax.get_figure().savefig('plot'+str(N) + '.png')


# Scatter plot of data points
plt.figure()
ax=sns.scatterplot(x     = x,
                   y     = y,
                   color = "blue",
                 )
ax.set(title   = "Data points")
#ax.get_figure().savefig('plot_data_'+str(N) + '.png')


# Using pre-programmed programs:
sm.OLS(y,X).fit().summary()