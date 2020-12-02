#!/usr/bin/env python
"""
MNIST data reader
"""
import struct
import numpy as np

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

def test() :
    for lab,arr in readVec('t10k') :
        print lab, "-------"
        show(arr)
        print

if __name__ == '__main__' :
    test()
