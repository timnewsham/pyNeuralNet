#!/usr/bin/env python
"""
neural net for mnist digit data
"""
from mynet import *
from mnist import *

net = mkNet(TanhNeuron, SquareCost, 784, 30, 10)

def getopt() :
    p = argparse.ArgumentParser(description='digit recognizer')
    p.add_argument('-D', dest='dfile', default='mnist/t10k')
    p.add_argument('-f', dest='nfile', default='mnist2.net')
    p.add_argument('-t', dest='train', action="store_true", default=False)
    p.add_argument('-e', dest='eps', type=float, default=3.0, help="epsilon for training")
    p.add_argument('-l', dest='loops', type=int, default=100)
    p.add_argument('-L', dest='lamb', type=float, default=0.1)
    p.add_argument('-b', dest='batch', type=int, default=100)
    p.add_argument('-B', dest='bloops', type=int, default=1)
    p.add_argument('-E', dest='enhance', action='store_true', default=False)
    opt = p.parse_args()
    return opt

def train(opt) :
    try :
        net.load(opt.nfile)
    except Exception, e :
        print 'cant load', e

    dat = [(img, mkDigitVec(lab)) for lab,img in readVec('mnist/t10k')]
    e0 = batchErr(net, dat, lamb=opt.lamb)
    print "initial error", e0
    try :
        batchLearn(net, opt.eps, bsz=opt.batch, nb=opt.bloops, nl=opt.loops, lamb=opt.lamb, dat=dat)
    except KeyboardInterrupt :
        print "interrupted"
        opt.nfile += "-int"
    print "initial error", e0
    print "  final error", batchErr(net, dat, lamb=opt.lamb)
    print "saving", opt.nfile
    net.save(opt.nfile)

def bestGuess(vec) :
    xs = list((p, idx) for idx,p in enumerate(vec))
    xs.sort()
    return xs[-1][1]

def ident(opt) :
    net.load(opt.nfile)

    # just try ten randomly for now
    dat = list(readVec(opt.dfile))
    random.shuffle(dat)
    cnt, ok = 0, 0
    for lab,img in dat[:100] :
        vec = net.fwd(img)
        showImgVec(lab, img, vec)
        if bestGuess(vec) == lab :
            ok += 1
        cnt += 1
    print "%d ok, %d total, %.1f%% correct" % (ok, cnt, ok * 100.0/cnt)

def enhance(opt) :
    net.load(opt.nfile)
    dat = list(readVec(opt.dfile))
    lab,img = random.choice(dat)
    vec = net.fwd(img)
    showImgVec(lab, img, vec)

    # use gradient to improve img for ideal output
    train = mkDigitVec(lab)
    cnt = 10000
    for m in xrange(cnt) :
        g = net.grad(img, train, retgX=True)
        img -= opt.eps * g
    vec = net.fwd(img)
    showImgVec(lab, img, vec)

def main() :
    opt = getopt()
    if opt.train :
        train(opt)
    elif opt.enhance :
        enhance(opt)
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
