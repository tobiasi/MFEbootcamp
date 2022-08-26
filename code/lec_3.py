#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:49:49 2021

@author: ruslebiffen
"""

from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import numpy             as np
import random as rand


"""
Riddle simulation
Let r denote the radius of a circle. Let Q be a uniformly distributed point 
within this circle, and x be the distance from the centre of the circle and to
the point Q. What is E[x]? What is Var[X]?
"""


x  = np.zeros(shape=(1000000,1))
y  = np.zeros(shape=(1000000,1))
ll = 0
for ii in range(0,10000):
    xx = 10*rand.uniform(-1,1)
    yy = 10*rand.uniform(-1,1)
    d  = xx**2 + yy**2
    if d<=100:
        x[ll] = xx
        y[ll] = yy
        ll    = ll + 1
        
x = x[0:ll]
y = y[0:ll]        
    
fig = plt.figure()
ax  = plt.axes()

ax.set_aspect(1)
theta = np.linspace(-np.pi, np.pi, 200)
plt.plot(np.sin(theta)*10, 10*np.cos(theta))
plt.scatter(x,y)
plt.show()


z = np.multiply(x,x)+np.multiply(y,y)
l = np.mean(np.sqrt(z))
k = np.var(np.sqrt(z))
print('\nE[x] = ' + str(l))
print('\nV[x] = ' + str(k))



'''
Fuzzy matching
'''



name1 = 'Microsoft'

name2 = 'microsoft'

name1 == name2


name1.lower() == name2


fuzz.ratio("UCLA","Anderson School of Management")

fuzz.ratio("UCLA","UCLA Anderson")


fuzz.ratio("applee inc","Apple Inc")


fuzz.ratio("Anderson School of Management","UCLA Anderson School of Management")

fuzz.partial_ratio("Anderson School of Management","UCLA Anderson School of Management")






namelist = [
        'Microsoft Corporation',
        'Oracle',
        'Tobias Corporation',
        'Microsoft',
        'Micro Corp',
        'Microosoft Corp',
        'MICROSOFT',
        'Apple Inc',
        'Tesla',
        'McDonalds',
        'Goldman Sachs']


score    = [np.nan]*len(namelist)
key      = 'microsoft'
low_list = [word.lower() for word in namelist]
for ind, item in enumerate(low_list):
    score[ind] = fuzz.partial_ratio(low_list[ind],key)
    
identified = [tok for tok, _ in np.flip(sorted(zip(score, namelist)))]


['Microsoft Corporation',
 'Microsoft',
 'MICROSOFT',
 'Microosoft Corp',
 'Micro Corp',
 'Tobias Corporation',
 'McDonalds',
 'Goldman Sachs',
 'Apple Inc',
 'Tesla',
 'Oracle']




'''
Path/folder management
'''

import os

main_path = '/Users/ruslebiffen/Documents/mfe'

os.listdir(main_path)

os.chdir(main_path)

os.listdir()

# Want to make a pointless folder
newpath = main_path + '/pointless_folder'

print(newpath)

print(newpath)

os.makedirs(newpath)

os.listdir()

os.chdir(newpath)

os.listdir()




























