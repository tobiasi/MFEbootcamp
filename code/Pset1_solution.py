#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 00:00:07 2022

@author: ruslebiffen
"""
import numpy as np


def magic_function(n):
    A = np.zeros((n,n))
    for ii in range(1,n+1):
        for jj in range(2, n  + 2 - ii):
            A[ii-1,jj+ii-2] = jj
            
    A = A + A.T + np.eye(n)
    
    return A


a = np.ones(5)
b = np.ones(6)
def vec_mult(a,b):
    if len(a) != len(b):
          raise Exception('The vectors must be of same length')
    dim_vec = len(a)
    ret     = 0    
    for ii in range(dim_vec):
        ret += a[ii]*b[ii]
    return ret        



vec_mult(a,b)

a = np.random.rand(5,5)
b = np.random.rand(5,5)


def multi(A,B):
    dim     = len(A)
    dim_A,_ = np.shape(A)
    dim_B,_ = np.shape(B)
    
    if dim_A != dim_B:
          raise Exception('The matrices must be of same size')
    
    C = np.zeros((dim_A,dim_A))
    
    for ii in range(dim):
        for jj in range(dim):
            a        = A[ii,:]
            b        = B[:,jj]
            C[ii,jj] = vec_mult(a,b)
    return C


