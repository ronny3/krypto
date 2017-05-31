#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing functions for data eg. normalize, moving average..
"""
import numpy as np
import pandas as pd
def Normalize(X_train):
    """ Normalizes all last dimensions"""
    for a in range(X_train.shape[-1]):
        for i in range(X_train.shape[0]):
            X_train[i,:,a] -= np.mean(X_train[i,:,a])
            X_train[i,:,a] /= np.std(X_train[i,:,a])
    
    return X_train
        
def MovingAverage(X_train, n=5):
    """ Transforms spiky data into moving average
    # Param n = averaging length
    """
    if n==1 or n==None:
        return X_train
    ret = np.cumsum(X_train, dtype='float64')
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
def pandaMean(X_train, n=5):
    a = np.random.rand(1000,1)
    df = pd.DataFrame(X_train[:,:,0],dtype='float64')
    #http://pandas.pydata.org/pandas-docs/stable/computation.html#moving-rolling-statistics-moments