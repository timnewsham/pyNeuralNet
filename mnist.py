#!/usr/bin/env python
"""
MNIST data reader
"""
import argparse, random, struct
import numpy as np

from mynet import Net, batchLearn, batchErr

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

def pixChar(p) :
    if p < 0.25 : return ' '
    if p < 0.50 : return '.'
    if p < 0.75 : return 'o'
    return '*'

def show(arr) :
    # assume 28x28
    for n in xrange(28) :
        row = arr[n*28 : (n+1)*28]
        print ''.join(pixChar(p) for p in row)

def showImgVec(lab, img, vec) :
    vlines = ['%d %02d%% %s' % (n, vec[n] * 100, '*' * int(vec[n]*10)) for n in xrange(10)]
    ilines = [''.join(pixChar(p) for p in img[n*28 : (n+1)*28]) for n in xrange(28)]
    ilines = ilines[4:-4]

    slines = []
    slines.append('label %d' % lab)
    slines.append('')

    best = [(prob,idx) for (idx,prob) in enumerate(vec)]
    best.sort(reverse=True)
    slines.append('best guesses:')
    for p,n in best[:3] :
        slines.append('%d %.2f%%' % (n, p))
    slines.append('')
    vlines = slines + vlines

    while len(vlines) < len(ilines) :
        vlines.append('')
    print '--------------'
    for v,i in zip(vlines, ilines) :
        print i, v


def mkDigitVec(n) :
    f = lambda m : 1.0 if m == n else 0.0
    return np.array([f(m) for m in xrange(10)])

def getopt() :
    p = argparse.ArgumentParser(description='digit recognizer')
    p.add_argument('-t', dest='train', action="store_true", default=False)
    p.add_argument('-e', dest='eps', type=float, default=0.1, help="epsilon for training")
    p.add_argument('-f', dest='nfile', default='mnist.net')
    p.add_argument('-l', dest='loops', type=int, default=100)
    p.add_argument('-b', dest='batch', type=int, default=100)
    p.add_argument('-B', dest='bsize', type=int, default=1000)
    opt = p.parse_args()
    return opt

def train(opt) :
    n = Net(784, 16, 16, 10)
    try :
        n.load(opt.nfile)
    except Exception, e :
        print 'cant load', e

    dat = [(img, mkDigitVec(lab)) for lab,img in readVec('mnist/t10k')]
    print "initial error", batchErr(n, dat)
    try :
        batchLearn(n, opt.eps, bsz=opt.batch, nb=opt.bsize, nl=opt.loops, dat=dat)
    except KeyboardInterrupt :
        print "interrupted"
        opt.nfile += "-int"
    print "final error", batchErr(n, dat)
    print "saving", opt.nfile
    n.save(opt.nfile)

def ident(opt) :
    net = Net(784, 16, 16, 10)
    net.load(opt.nfile)

    # just try ten randomly for now
    dat = list(readVec('mnist/t10k'))
    random.shuffle(dat)
    for lab,img in dat[:10] :
        vec = net.fwd(img)
        showImgVec(lab, img, vec)

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
