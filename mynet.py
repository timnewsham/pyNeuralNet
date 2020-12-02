#!/usr/bin/env python
"""
Neural network with backprop training.
"""
from math import exp
import numpy as np

def sig(x) :
    return 1 / (1 + np.exp(-x))

def dsig(x) :
    s = sig(x)
    return s * (1 - s)

class Net(object) :
    def __init__(self, *size) :
        """Size should be a list of number of nodes in each layer."""
        self.W = []
        self.B = []
        for prev,cur in zip(size, size[1:]) :
            self.W.append( np.random.randn(cur, prev) )
            self.B.append( np.random.randn(cur) )

    def fwd(self, I) :
        X = I
        self.X = [X]
        self.Z = []
        for W,B in zip(self.W, self.B) :
            Z = np.dot(W, X) + B
            X = sig(Z)
            self.X.append(X)
            self.Z.append(Z)
        return X

    def err(self, I, T) :
        O = self.fwd(I)
        E = O - T
        return np.dot(E, E)

    def grad(self, I, T) :
        O = self.fwd(I)

        # denote  d(ERR)/dX = gX
        gB = []
        gW = []
        levels = len(self.W)
        gZ = None
        for rev in xrange(1, levels+1) :
            if gZ is None :
                # start off with gX from output error
                gX = 2 * (O - T)
            else :
                # use previous level gZ to compute gX
                Wprev = self.W[-rev + 1]
                gX = np.dot(Wprev.T, gZ)
            gZ = gX * dsig(self.Z[-rev])

            # accumulate gB/gW in reverse
            gB.insert(0, gZ)
            gW.insert(0, np.outer(gZ, self.X[-rev - 1]))
        return gB, gW

    def update(self, eps, gB, gW) :
        for l in xrange(2) :
            self.B[l] -= eps * gB[l]
            self.W[l] -= eps * gW[l]

def testTrain(n, i, t) :
    if 1 : # back prop
        for m in xrange(10000) :
            gB,gW = n.grad(i, t)
            n.update(0.001, gB, gW)
            if m % 1000 == 0 :
                print m, n.err(i, t)

    print "trained for", t
    print "forward", n.fwd(i)
    print "err", n.err(i, t)

def test() :
    if 0 :
        n = Net(2, 2, 2)
        i = np.array([2,4])
        t = np.array([0.5, 0.1])
        testTrain(n, i, t)

    if 1 :
        n = Net(2,3,1)
        i = np.array([2,4])
        t = np.array([0.5])
        testTrain(n, i, t)

if __name__ == '__main__' :
    test()
