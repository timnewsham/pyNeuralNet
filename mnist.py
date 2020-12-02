#!/usr/bin/env python
"""
MNIST data reader
"""
import argparse, random, struct
import numpy as np

from mynet import Net

class Error(Exception) :
    pass

def reader(basename) :
    fnImg = basename + "-images-idx3-ubyte"
    fnIdx = basename + "-labels-idx1-ubyte"
    fImg = file(fnImg, 'rb')
    fIdx = file(fnIdx, 'rb')
    mag,cnt = struct.unpack("!II", fIdx.read(8))
    if mag != 0x801 :
        raise Error("bad magic for " + fnIdx)
    mag,cnt2,h,w = struct.unpack("!IIII", fImg.read(16))
    if mag != 0x803 :
        raise Error("bad magic for " + fnImg)
    if cnt != cnt2 :
        raise Error("index and image files have different sizes")

    for n in xrange(cnt) :
        img = map(ord, fImg.read(w*h))
        lab = ord(fIdx.read(1))
        yield lab,img

def readVec(basename) :
    for lab,img in reader(basename) :
        arr = np.array([pix/255.0 for pix in img])
        yield lab, arr

def readImg(basename) :
    # XXX convert into some displayable format to return
    pass

def show(arr) :
    def pixChar(p) :
        if p > 0.5 :
            return '*'
        return ' '

    # assume 28x28
    for n in xrange(28) :
        row = arr[n*28 : (n+1)*28]
        print ''.join(pixChar(p) for p in row)

def mkDigitVec(n) :
    f = lambda m : 1.0 if m == n else 0.0
    return np.array([f(m) for m in xrange(10)])

def batchLearn(net, eps, bsz, nb, nl, dat) :
    vec = [mkDigitVec(n) for n in xrange(10)]
    for lcnt in xrange(nl) :
        random.shuffle(dat)
        batch = dat[:bsz]
        for bcnt in xrange(nb) :
            # average gradient over training batch
            GB,GW = None,None
            for lab,img in batch :
                gB,gW = net.grad(img, vec[lab])
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

        if 1 :
            err = sum(net.err(img, vec[lab]) for lab,img in batch) / nb
            print 'avg batch error', err

def getopt() :
    p = argparse.ArgumentParser(description='digit recognizer')
    p.add_argument('-t', dest='train', action="store_true", default=False)
    p.add_argument('-e', dest='eps', type=float, default=0.00001, help="epsilon for training")
    p.add_argument('-f', dest='nfile', default='mnist.net')
    opt = p.parse_args()
    return opt

def train(opt) :
    n = Net(784, 16, 16, 10)
    try :
        n.load(opt.nfile)
    except Exception, e :
        print 'cant load', e

    dat = list(readVec('mnist/t10k'))
    try :
        batchLearn(n, opt.eps, bsz=100, nb=1000, nl=10, dat=dat)
    except KeyboardInterrupt :
        print "interrupted"
        opt.nfile += "-int"
    print "saving", opt.nfile
    n.save(opt.nfile)

def ident(opt) :
    net = Net(784, 16, 16, 10)
    net.load(opt.nfile)

    # just try ten randomly for now
    dat = list(readVec('mnist/t10k'))
    random.shuffle(dat)
    for lab,img in dat[:10] :
        show(img)
        best = [(prob,idx) for (idx,prob) in enumerate(net.fwd(img))]
        best.sort(reverse=True)
        print "label", lab
        for p,n in best[:3] :
            print n, '%%%1.f' % (100.0 * p)

def main() :
    opt = getopt()
    if opt.train :
        train(opt)
    else :
        ident(opt)

def test() :
    for lab,arr in readVec('mnist/t10k') :
        print lab, "-------"
        show(arr)
        print

if __name__ == '__main__' :
    #test()
    main()
