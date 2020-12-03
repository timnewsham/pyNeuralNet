#!/usr/bin/env python
"""
Neural network with backprop training.
"""
from math import exp
import numpy as np
import copy, pickle, random

class SigmoidNeuron(object) :
    def neuronFunc(self, x) :
        """Sigmoid neuron function."""
        return 1 / (1 + np.exp(-x))
    def dNeuronFunc(self, x) :
        """Derivative of sigmoid neuron function."""
        s = self.neuronFunc(x)
        return s * (1 - s)

class SquareCost(object) :
    def costFunc(self, O, T) :
        """Square error function."""
        E = O - T
        return np.dot(E, E)
    def dCostFunc(self, O, T) :
        """Derivative of square error function."""
        return 2 * (O - T)


class NetBase(object) :
    def __init__(self, *size) :
        """Size should be a list of number of nodes in each layer."""
        self.W = []
        self.B = []
        for prev,cur in zip(size, size[1:]) :
            self.W.append( np.random.randn(cur, prev) )
            self.B.append( np.random.randn(cur) )

    def save(self, fn) :
        dat = (self.W, self.B)
        pickle.dump(dat, file(fn, 'wb'))
    def load(self, fn) :
        dat = pickle.load(file(fn, 'r'))
        self.W, self.B = dat

    def fwd(self, I) :
        X = I
        self.X = [X]
        self.Z = []
        for W,B in zip(self.W, self.B) :
            Z = np.dot(W, X) + B
            X = self.neuronFunc(Z)
            self.X.append(X)
            self.Z.append(Z)
        return X

    def err(self, I, T) :
        O = self.fwd(I)
        return self.costFunc(O, T) / len(O)

    def grad(self, I, T, retgX=False) :
        O = self.fwd(I)

        # denote  d(ERR)/dX = gX
        gB = []
        gW = []
        levels = len(self.W)
        gZ = None
        for rev in xrange(1, levels+1) :
            if gZ is None :
                # start off with gX from output error
                gX = self.dCostFunc(O, T)
            else :
                # use previous level gZ to compute gX
                Wprev = self.W[-rev + 1]
                gX = np.dot(gZ.T, Wprev)
            gZ = gX * self.dNeuronFunc(self.Z[-rev])

            # accumulate gB/gW in reverse
            gB.insert(0, gZ)
            gW.insert(0, np.outer(gZ, self.X[-rev - 1]))
        if retgX :
            gX = np.dot(gZ.T, self.W[0])
            return gX
        return gB, gW

    def update(self, eps, gB, gW) :
        for l in xrange(2) :
            self.B[l] -= eps * gB[l]
            self.W[l] -= eps * gW[l]

class Net(NetBase, SigmoidNeuron, SquareCost) :
    pass

def batchErr(net, dat) :
    return sum(net.err(inp, outp) for inp,outp in dat) / len(dat)

def batchLearn(net, eps, bsz, nb, nl, dat, verbose=True) :
    if verbose :
        print "eps %f batch size %d batch iters %d loops %d" % (eps, bsz, nb, nl)
    for lcnt in xrange(nl) :
        random.shuffle(dat)
        batch = dat[:bsz]
        for bcnt in xrange(nb) :
            # average gradient over training batch
            GB,GW = None,None
            for inp,outp in batch :
                gB,gW = net.grad(inp, outp)
                if GB is None :
                    GB, GW = gB,gW
                else :
                    for x,y in zip(GB, gB) :
                        x += y
                    for x,y in zip(GW, gW) :
                        x += y
            for x in GB :
                x *= (1.0 / nb)
            for x in GW :
                x *= (1.0 / nb)

            net.update(eps, GB, GW)

        if verbose :
            print 'avg batch error', batchErr(net, batch)


def testGrads(eps, N, I, T) :
    """Helper to see if grads are sane by comparing to numeric estimates."""
    def gradB(l, i) :
        n = copy.deepcopy(N)
        n.B[l][i] -= eps
        return (e0 - n.err(I, T)) / eps
    def gradW(l, i, j) :
        n = copy.deepcopy(N)
        n.W[l][i,j] -= eps
        return (e0 - n.err(I, T)) / eps
    e0 = N.err(I, T)
    gBs, gWs = N.grad(I, T)
    for l,(gB, gW) in enumerate(zip(gBs, gWs)) :
        n,m = gW.shape
        gB2 = np.array([gradB(l, i) for i in xrange(n)])
        gW2 = np.array([[gradW(l,i,j) for j in xrange(m)] for i in xrange(n)])
        print 'err B', l, (gB2 - gB)
        print 'err W', l, (gW2 - gW)

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

    if 0 :
        n = Net(2,3,1)
        i = np.array([2,4])
        t = np.array([0.5])
        testTrain(n, i, t)

    if 0 :
        n = Net(2,3,4,1)
        i = np.array([2,4])
        t = np.array([0.5])
        print n.err(i, t)
        testGrads(0.0000001, n, i, t)

    if 1 :
        n = Net(2,3,4,1)
        i = np.array([2,4])
        print n.fwd(i)

        n.save("test-save")
        m = Net(2,3,4,1)
        m.load("test-save")
        print m.fwd(i)

if __name__ == '__main__' :
    test()
