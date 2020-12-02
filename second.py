#!/usr/bin/env python
"""
Try backprop of a 2 layer net
using sigmoid.
"""
from math import exp
import numpy as np

def sig(x) :
    return 1 / (1 + np.exp(-x))

def dsig(x) :
    s = sig(x)
    return s * (1 - s)

def sigb(o) :
    """partial of o = sig(x) for x, given o."""
    return o * (1 - o)

class Net(object) :
    def __init__(self) :
        self.W = [np.random.randn(2,2), np.random.randn(2,2)]
        self.B = [np.random.randn(2), np.random.randn(2)]
    def fwd(self, I) :
        W1,W2 = self.W
        B1,B2 = self.B
        X0 = I
        Z1 = np.dot(W1, X0) + B1
        X1 = sig(Z1)
        Z2 = np.dot(W2, X1) + B2
        X2 = sig(Z2)
        self.X = [X0, X1, X2]
        self.Z = [Z1, Z2]
        return X2

    def fwd2(self, I) :
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
        X0,X1,X2 = self.X
        Z1,Z2 = self.Z
        W1,W2 = self.W
        B1,B2 = self.B

        # denote  d(ERR)/dX = gX
        E = X2 - T
        gX2 = 2 * E
        gZ2 = gX2 * dsig(Z2)
        gB2 = gZ2
        gW2 = np.outer(gB2, X1)

        gX1 = np.dot(W2.T, gZ2)
        print 'gX1', gX1
        gZ1 = gX1 * dsig(Z1)
        gB1 = gZ1
        gW1 = np.outer(gB1, X0)

        return (gB1, gB2), (gW1, gW2)

    def grad2(self, I, T) :
        # denote  d(ERR)/dX = gX
        gB = []
        gW = []
        levels = len(self.W)
        gZ = None
        for rev in xrange(1, levels+1) :
            if gZ is None :
                # start off with gX from output error
                gX = 2 * (self.X[-rev] - T)
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

    def gradNum(self, I, T) :
        eps = 0.000001
        e0 = self.err(I, T)
        def gradB(l, j) :
            t = self.B[l][j]
            self.B[l][j] -= eps
            d = (e0 - self.err(I, T)) / eps
            self.B[l][j] = t
            return d
        def gradW(l, j, k) :
            t = self.W[l][j,k]
            self.W[l][j,k] -= eps
            d = (e0 - self.err(I, T)) / eps
            self.W[l][j,k] = t
            return d

        gB = [np.array([gradB(l, i) for i in xrange(2)]) for l in xrange(2)]
        gW = [np.array([[gradW(l,i,j) for j in xrange(2)] for i in xrange(2)]) for l in xrange(2)]
        return gB, gW

def test() :
    n = Net()
    i = np.array([2,4])
    t = np.array([0.5, 0.5])
    x = n.fwd(i)
    print n.err(i, t)

    eps = 0.01
    if 0 : # numeric back prop
        for m in xrange(100) :
            gB,gW = n.gradNum(i, t)
            n.update(0.01, gB, gW)

    if 1 : # analytic back prop
        for m in xrange(100) :
            gB,gW = n.grad2(i, t)
            n.update(0.01, gB, gW)

    if 0 : # numeric grads vs analytic grads
        GB,GW = n.gradNum(i, t)
        gB,gW = n.grad(i, t)
        for l in xrange(2) :
            print 'level', l+1
            print 'num B',
            print GB[l]
            print 'ana B',
            print gB[l]
            print

            print 'num W',
            print GW[l]
            print 'ana W',
            print gW[l]
            print

    if 0 : # numeric grads vs analytic grads
        GB,GW = n.grad(i, t)
        gB,gW = n.grad2(i, t)
        for l in xrange(2) :
            print 'level', l+1
            print 'ana1 B',
            print GB[l]
            print 'ana2 B',
            print gB[l]
            print

            print 'ana1 W',
            print GW[l]
            print 'ana2 W',
            print gW[l]
            print

    print n.err(i, t)

    if 1 : # compare fwd/fwd2 for equality
        print 'fwd1', n.fwd(i)
        print 'fwd2', n.fwd2(i)


if __name__ == '__main__' :
    test()
