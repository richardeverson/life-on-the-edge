#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
====================================
Find the edges in non-dominated sets
====================================


:Author:
   Richard Everson   <R.M.Everson@exeter.ac.uk>
:Date:
   7 September 2012
:Copyright:
   Copyright (c)  Richard Everson, University of Exeter, 2012
:File:
   edge.py
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import numpy.random.mtrand
import scipy.misc
import scipy.stats.mstats
import sys

def rankCoords(X):
    """
    Return X in rank coordinates
    """
    Z = np.zeros(X.shape)
    for c in range(X.shape[1]):
        Z[:,c] = scipy.stats.mstats.rankdata(X[:,c])
    return Z


def nondom(Z):
    """
    Return a list of the indices of the non-dominated rows of Z.
    """
    N, D = Z.shape
    
    nd = []
    for n in range(N):
        # First term finds rows Z[j,:] which are smaller than or equal to
        # Z[n,:]; second term checks that there is at least one element of
        # Z[n] strictly smaller.
        dominators = np.sum( (np.sum(Z <= Z[n], 1) == D) * (np.sum(Z < Z[n], 1) > 0) )
        if dominators == 0:
            nd.append(n)
    return nd

def nondom_max(Z):
    """
    Return a list of the indices of the rows of Z that are non-dominated under maximimisation.
    """
    N, D = Z.shape
    
    nd = []
    for n in range(N):
        # First term finds rows Z[j,:] which are greater than or equal to
        # Z[n,:]; second term checks that there is at least one element of
        # Z[n] strictly greater.
        dominators = np.sum( (np.sum(Z >= Z[n], 1) == D) * (np.sum(Z > Z[n], 1) > 0) )
        if dominators == 0:
            nd.append(n)
    return nd


def edges(Y, k=None, solo=False, max=False):
    """
    Find the edges in Y by projection onto k-dimensional subsets. If k is None
    corners of all orders are found; if k is specified only edges of the given
    order k are located.

    Returns a list of the indices of Y that are edge points

    If max is True the points found are non-dominated under maximisation
    rather than minimisation.
    
    If solo is True, a point is an edge point if it is the sole dominator in a
    particular projection; this is similar to Singh et al's "corners". If
    False all the non-dominated points in that projection are edges.
    """ 
    ndfunc = nondom_max if max else nondom
   
    N, M = Y.shape
    if k is None:
        K = range(1, M)
    else:
        K = [k]
    
    corners = {}
    for k in K:
        print k
        for criteria in combinations(range(M), k):
            sys.stdout.flush()
            Yk = Y[:, criteria]
            nd = ndfunc(Yk)
            if solo and len(nd) == 1:
                if nd[0] not in corners:
                    print k, nd[0]
                corners[nd[0]] = 1
            elif not solo:
                for n in nd:
                    corners[n] = 1
                    
    return corners.keys()



def weak_dominates(Y, x):
    """
    c = weak_dominates(Y, x)
    Return a vector signifying whether each row of Y weakly dominates x
    """
    return (Y <= x).sum(axis=1) == Y.shape[1]


def attain_edge(Y, detailed=False):
    """
    Return indices of rows of Y that are edge points of Y in the sense that
    they extend the range of the attainment surface.

    If detailed is True a list of tuples is returned, the first element of
    the tuple is the index i of the edge point and the second is the
    dimension in which extending the point to rplus produces a new point
    that is dominated only by Y[i]
    """

    rplus = Y.max(axis=0) + 1;

    edge = []
    for i, y in enumerate(Y):

        for m in range(Y.shape[1]):
            x = y.copy()
            x[m] = rplus[m]

            doms = weak_dominates(Y, x).sum()
            assert doms > 0

            if  doms == 1:          # Exactly one dominator, therefore edge
                if detailed:
                    edge.append((i, m))
                else:
                    edge.append(i)
                    break            # Don't bother testing this y any more
        else:
            pass
            #print 'Not weakly dominated!', y, x
    return edge

def dominance_distance(X, w=None):
    """
    Find the dominance distance between pairs of rows of X. Objectives
    (columns) are weighted by w. If w is None (the default) then the weights
    are all unity.

    Returns a matrix of the dominance distances.
    """
    N, M = X.shape
    if w is None:
        w = np.ones(M)
        
    R = np.zeros((N,M))
    for m in range(M):
        R[:,m] = scipy.stats.mstats.rankdata(X[:,m])
    D = np.zeros((N,N))
    for m in range(M):
        r = np.tile(R[:,m], (N,1))
        D += w[m]*np.abs(r - r.T)
    D /= M
    return D


def edge_frequency(Y, k=2):
    """
    Find the fraction of k-dimensional subset projections for which each point in Y is
    non-dominated under maximisation or minimisation.

    Returns an array of the fractions.
    """
    N, M = Y.shape
    f = np.zeros(N)
    
    for criteria in combinations(range(M), k):
        Yk = Y[:, criteria]
        nd = []
        for ndfunc in (nondom, nondom_max):
            nd += ndfunc(Yk)
        nd = list(set(nd))                          # Unique elements
        f[nd] += 1
    return f/(scipy.misc.comb(M, k, exact=1))


def edge_frequency_detailed(Y, k=2):
    """
    Find the fraction of k-dimensional subset projections for which each point in Y is
    non-dominated under minimisation, maximisation or both
    
    Returns arrays of the fractions: min, max, both.
    """
    N, M = Y.shape
    f = np.zeros(N)
    fmin = np.zeros(N)
    fmax = np.zeros(N)
    
    for criteria in combinations(range(M), k):
        Yk = Y[:, criteria]
        ndmin = nondom(Yk)
        fmin[ndmin] += 1
        ndmax = nondom_max(Yk)
        fmax[ndmax] += 1
        nd = list(set(ndmin+ndmax))                 # Unique elements
        f[nd] += 1
        
    MCk = scipy.misc.comb(M, k, exact=1)
    return fmin/MCk, fmax/MCk, f/MCk

def freq_hist(E, F, bounds, Nx = 30):
    """
    Histogram the frequencies
    """
    d = (bounds[1] - bounds[0])/Nx
    Ny = int(np.ceil((bounds[3]-bounds[2])/d))

    x = F[:,0]
    y = F[:,1]

    xloc = bounds[0] + np.arange(Nx+1)*d
    yloc = bounds[2] + np.arange(Ny+1)*d
    H = np.zeros((Nx, Ny))
    
    for i in range(Nx):
        for j in range(Ny):
            Ix = np.logical_and(xloc[i] < x, x <= xloc[i+1])
            Iy = np.logical_and(yloc[j] < y, y <= yloc[j+1])
            I = np.logical_and(Ix, Iy)
            if any(I):
                H[i,j] = np.median(E[I])
            else:
                H[i,j] = -1


    xloc[-1] -= 0.002
    yloc[-1] -= 0.002
    return H, xloc, yloc

