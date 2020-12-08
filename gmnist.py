#!/usr/bin/env python

from mnist import *

def getopt() :
    p = argparse.ArgumentParser(description='digit recognizer')
    p.add_argument('-f', dest='nfile', default='gmnist.net')
    p.add_argument('-t', dest='train', action="store_true", default=False)
    p.add_argument('-e', dest='eps', type=float, default=0.1, help="epsilon for training")
    p.add_argument('-l', dest='loops', type=int, default=100)
    p.add_argument('-b', dest='batch', type=int, default=100)
    p.add_argument('-B', dest='bsize', type=int, default=1000)
    opt = p.parse_args()
    return opt

def train(opt) :
    n = Net(10, 16, 16, 784)
    try :
        n.load(opt.nfile)
    except Exception, e :
        print 'cant load', e

    dat = [(mkDigitVec(lab), img) for lab,img in readVec('mnist/t10k')]
    e0 = batchErr(n, dat)
    print "initial error", e0
    try :
        batchLearn(n, opt.eps, bsz=opt.batch, nb=opt.bsize, nl=opt.loops, dat=dat)
    except KeyboardInterrupt :
        print "interrupted"
        opt.nfile += "-int"
    print "initial error", e0
    print "  final error", batchErr(n, dat)
    print "saving", opt.nfile
    n.save(opt.nfile)

def gen(opt) :
    net = Net(10, 16, 16, 784)
    net.load(opt.nfile)

    lab = random.randrange(0,10)
    vec = mkDigitVec(lab)
    img = net.fwd(vec)
    showImgVec(lab, img, vec)

def main() :
    opt = getopt()
    if opt.train :
        train(opt)
    else :
        gen(opt)

def test() :
    for lab,arr in readVec('mnist/t10k') :
        print lab, "-------"
        show(arr)
        print

if __name__ == '__main__' :
    #test()
    main()
