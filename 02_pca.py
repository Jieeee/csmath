# PCA

# load and convert data
import numpy as np


def load(file):
    f = open(file)
    line_number = 0

    while line_number < 21:
        line = f.readline()
        if line.startswith('entwidth'):
            segs = line.split(' ')
            w = eval(segs[-1])
        elif line.startswith('entheight'):
            segs = line.split(' ')
            h = eval(segs[-1])
        elif line.startswith('ndigit'):
            segs = line.split(' ')
            sz = eval(segs[-1])
        line_number += 1
    X = np.zeros([h, w, sz], dtype='uint8')
    Y = np.zeros(sz, dtype='uint8')
    digits = []
    for i in range(sz):
        digit = []
        for j in range(32):
            line = f.readline()
            row = [eval(v) for v in line[0:-1]]
            digit.append(row)

        X[..., i] = np.array(digit)
        Y[i] = eval(f.readline())

    return X, Y


if __name__ == '__main__':

    X ,Y = load('D:\\Course\\csmath\\hw2\\optdigits-orig.tra')

    print('Y[2] = ', Y[2]);
    for i in range(32):
        print(X[i, : , 2])

    print(X.shape)





