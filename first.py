#!/usr/bin/env python
"""
First try at backprop, single layer.
using sigmoid.
"""
from math import exp
import numpy as np

def sig(x) :
    return 1 / (1 + np.exp(-x))

def sigb(o) :
    """partial of o = sig(x) for x, given o."""
    return o * (1 - o)

class Net(object) :
    def __init__(self) :
        self.W = np.random.randn(2,2)
        self.B = np.random.randn(2)
    def fwd(self, I) :
        self.I = I
        self.O = sig(np.dot(self.W, self.I) + self.B)
        return self.O
    def err(self, I, T) :
        self.fwd(I)
        self.E = self.O - T
        return np.dot(self.E, self.E)
    def grad(self, I, T) :
        # de/d(Bi)  = 2 * (Oi - Ti) * Oi * (1 - Oi)
        #           = 2 * Ei * sigb(Oi)
        # de/d(Wij) = 2 * (Oi - Ti) * Oi * (1 - Oi) * Ij
        #           = 2 * Ei * sigb(Oi) * Ij
        O = self.fwd(I)
        E = O - T
        GB = 2 * E * sigb(O)
        GW = np.outer(GB, I)
        return GB, GW
    def gradNum(self, I, T) :
        eps = 0.00001
        e0 = self.err(I, T)
        def gradB(i) :
            t = self.B[i]
            self.B[i] -= eps
            d = (e0 - self.err(I, T)) / eps
            self.B[i] = t
            return d
        def gradW(i, j) :
            t = self.W[i,j]
            self.W[i,j] -= eps
            d = (e0 - self.err(I, T)) / eps
            self.W[i,j] = t
            return d
        GB = np.array([gradB(i) for i in xrange(2)])
        GW = np.array([[gradW(i,j) for i in xrange(2)] for j in xrange(2)])
        return GB, GW

def test() :
    n = Net()
    i = np.array([2,4])
    t = np.array([0.5, 0.5])
    x = n.fwd(i)
    print n.err(i, t)

    if 0 : # numeric back prop
        for m in xrange(100) :
            GB,GW = n.gradNum(i, t)
            n.B -= 0.1 * GB
        n.W -= 0.1 * GW

    if 1 : # analytic back prop
        for m in xrange(100) :
            GB,GW = n.grad(i, t)
            n.B -= 0.1 * GB
        n.W -= 0.1 * GW

    if 0 : # numeric grads vs analytic grads
        GB,GW = n.gradNum(i, t)
        gB,gW = n.grad(i, t)
        print GB
        print gB
        print GW
        print gW

    print n.err(i, t)

if __name__ == '__main__' :
    test()
