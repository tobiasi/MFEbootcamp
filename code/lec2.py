#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:47:04 2022

@author: ruslebiffen
"""

from numpy.linalg import matrix_rank
import numpy as np

A = np.array([[2, 4, -1, 3], [-4, -3, 9, 5] , [6, 12, -3, 9]])

A

matrix_rank(A)


B = np.eye(3)
B

matrix_rank(B) 


C = np.array([[2, 4, -1, 3], [-4, -3, 9, 5]])


C

C.T


D = np.array(np.random.randint(0,10, size=(3, 3)))
E = np.array(np.random.randint(0,10, size=(3, 3)))



np.matmul(D,E)
# or 
D@E



E@D


F = np.array(np.random.randint(0,10, size=(3, 3)))
G = np.eye(3)


F

G

G @ F


H = np.matrix(np.random.randint(0,10, size=(3, 3))) 
I = np.linalg.inv(H)


H@I


A = np.array([[1, 3, 4], [2,-1,3] , [3,2,5]])
b = np.array([11,3,12])


np.linalg.inv(A) @ b



