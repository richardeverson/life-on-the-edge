#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
=======================================================
Hourglass -- generate hourglass data in many dimensions
=======================================================


:Author:
         Richard Everson <R.M.Everson@exeter.ac.uk>
:Date:
         5 October 2013 
:Copyright:
         Copyright (c) Richard Everson, University of Exeter, 2013
:File:
         hourglass.py
"""

from __future__ import division
import numpy as np

def mkhourglass(D=3, N=500):
    """
    Make an hourglass data set in D dimensions with N samples
    """
    
    N, M = N//2, (N+1)//2
    if np.random.randn(1) < 0.0:
        N, M = M, N
        
    X = np.abs(np.random.randn(N, D))
    for j in range(N):
        X[j] = X[j]/np.linalg.norm(X[j])
            
    d = np.ones(D)
    d[-1] = 0
    X += d
            
    Y = np.abs(np.random.randn(M, D))
    for j in range(M):
        Y[j] = Y[j]/np.linalg.norm(Y[j])
                
    Y = np.ones(D) - Y
    d = np.zeros(D)
    d[-1] = 1
    Y += d
                
    Y = np.vstack((X, Y))
    return Y
