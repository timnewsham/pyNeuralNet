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

class RectLinNeuron(object) :
    def neuronFunc(self, x) :
        return np.maximum(x, 0)
    def dNeuronFunc(self, x) :
        # heaviside == step function, 1 if x > 0.
        return np.heaviside(x, 0)

class TanhNeuron(object) :
    def neuronFunc(self, x) :
        return np.tanh(x)
    def dNeuronFunc(self, x) :
        sech = 1.0 / np.cosh(x)
        return sech ** 2

class SquareCost(object) :
    def costFunc(self, O, T) :
        """Square error function."""
        E = O - T
        return 0.5 * np.dot(E, E)
    def dCostFunc(self, O, T) :
        """Derivative of square error function."""
        return O - T

class CrossEntropyCost(object) :
    # Note: tanh output range is [-1,1] and will not work with
    # this cost function, which expects inputs from [0,1].
    def costFunc(self, O, T) :
        C = -T * np.log(O) - (1.0 - T) * np.log(1.0 - O)
        #assert not any(x < 0 for x in O)
        #assert not any(x > 1 for x in O)
        return np.sum(np.nan_to_num(C)) 
    def dCostFunc(self, O, T) :
        return (O - T) / (O - O*O)

class NetBase(object) :
    def __init__(self, *size) :
        """Size should be a list of number of nodes in each layer."""
        self.W = []
        self.B = []
        for prev,cur in zip(size, size[1:]) :
            # random W's scaled back by number of inputs to avoid saturation
            self.W.append( np.random.randn(cur, prev) / np.sqrt(prev) )
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

    def err(self, I, T, lamb=0) :
        O = self.fwd(I)
        cost = self.costFunc(O, T) / len(O)
        regCost = cost + 0.5 * lamb * sum(np.sum(W*W) for W in self.W)
        return regCost

    def grad(self, I, T, lamb=0, retgX=False) :
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
            # lambda term is for regularization
            dW = np.outer(gZ, self.X[-rev - 1]) + lamb * self.W[-rev]
            gW.insert(0, dW)
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

def mkNet(neuron, cost, *size) :
    class NN(NetBase, neuron, cost) :
        pass
    return NN(*size)

def batchErr(net, dat, lamb=0) :
    return sum(net.err(inp, outp, lamb=lamb) for inp,outp in dat) / len(dat)

def batchLearn(net, eps, bsz, nb, nl, dat, lamb=0, verbose=True) :
    # XXX vectorize this for all training data in batch
    if verbose :
        print "eps %f lambda %f batch size %d batch iters %d loops %d" % (eps, lamb, bsz, nb, nl)
    for lcnt in xrange(nl) :
        random.shuffle(dat)
        batch = dat[:bsz]
        for bcnt in xrange(nb) :
            # average gradient over training batch
            GB,GW = None,None
            for inp,outp in batch :
                gB,gW = net.grad(inp, outp, lamb=lamb)
                if GB is None :
                    GB, GW = gB,gW
                else :
                    for x,y in zip(GB, gB) :
                        x += y
                    for x,y in zip(GW, gW) :
                        x += y
            for x in GB :
                x *= (1.0 / (bsz * nb))
            for x in GW :
                x *= (1.0 / (bsz * nb))

            net.update(eps, GB, GW)

        if verbose :
            print 'avg batch error', batchErr(net, batch, lamb=lamb)


def testGrads(eps, N, I, T, lamb=0) :
    """Helper to see if grads are sane by comparing to numeric estimates."""
    def gradB(l, i) :
        n = copy.deepcopy(N)
        n.B[l][i] -= eps
        return (e0 - n.err(I, T, lamb=lamb)) / eps
    def gradW(l, i, j) :
        n = copy.deepcopy(N)
        n.W[l][i,j] -= eps
        return (e0 - n.err(I, T, lamb=lamb)) / eps
    e0 = N.err(I, T, lamb=lamb)
    gBs, gWs = N.grad(I, T, lamb=lamb)
    for l,(gB, gW) in enumerate(zip(gBs, gWs)) :
        n,m = gW.shape
        gB2 = np.array([gradB(l, i) for i in xrange(n)])
        gW2 = np.array([[gradW(l,i,j) for j in xrange(m)] for i in xrange(n)])
        print 'ratio B to numeric', l, (gB / gB2)
        print 'ratio W to numeric', l, (gW / gW2)

def testTrain(n, i, t, lamb=0) :
    print "initial output", n.fwd(i)
    print "initial error", n.err(i, t, lamb=lamb)
    for m in xrange(10000) :
        gB,gW = n.grad(i, t, lamb=lamb)
        n.update(0.001, gB, gW)
        if m % 1000 == 0 :
            print m, n.err(i, t, lamb=lamb)

    print "trained for", t
    print "forward", n.fwd(i)
    print "err", n.err(i, t, lamb=lamb)

def testAllGrads(lamb=0) :
    """Check gradients for all combos of cost and neuron."""
    for neuron in SigmoidNeuron, TanhNeuron, RectLinNeuron :
        for cost in SquareCost, CrossEntropyCost :
            if neuron == RectLinNeuron :
                continue # skip for now
            if neuron == TanhNeuron and cost == CrossEntropyCost :
                continue # combo doesnt make sense
            net = mkNet(neuron, cost, 2,3,4,1)
            print "testing", neuron, cost
            i = np.array([2,4])
            t = np.array([0.5])
            print net.err(i, t)
            testGrads(0.0000001, net, i, t, lamb=lamb)
            print

def test() :
    if 1 :
        #net = mkNet(SigmoidNeuron, SquareCost, 2, 2, 2)
        #net = mkNet(SigmoidNeuron, CrossEntropyCost, 2, 2, 2)
        #net = mkNet(TanhNeuron, CrossEntropyCost, 2, 2, 2) # XXX!
        net = mkNet(TanhNeuron, SquareCost, 2, 2, 2)
        i = np.array([2,4])
        t = np.array([0.5, 0.1])
        testTrain(net, i, t, lamb=0.5)

    if 0 :
        n = Net(2,3,1)
        i = np.array([2,4])
        t = np.array([0.5])
        testTrain(n, i, t)

    if 0 :
        testAllGrads()
    if 0 :
        testAllGrads(lamb=0.1)

    if 0 :
        net = Net(2,3,4,1)
        i = np.array([2,4])
        t = np.array([0.5])
        testGrads(0.0000001, net, i, t, lamb=0.5)

    if 0 :
        n = Net(2,3,4,1)
        i = np.array([2,4])
        print n.fwd(i)

        n.save("test-save")
        m = Net(2,3,4,1)
        m.load("test-save")
        print m.fwd(i)

if __name__ == '__main__' :
    test()
