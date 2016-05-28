# PCA

# load and convert data
import numpy as np
import matplotlib.pyplot as plt
#import skimage.io as io
import skimage.transform as tf

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

    X, Y = load('D:\\Course\\csmath\\hw2\\optdigits-orig.tra')

    # dimension reduction simply using resize function
    X_resize = tf.resize(X, [16, 16])

    # Logic vector
    logic_v = (Y == 3)
    three_data = X_resize[..., logic_v]
    three_origin = X[..., logic_v]
    print('sum:', sum(logic_v))
    print(three_data.shape)
    three_count = three_data.shape[2]
    three_vec = np.reshape(three_data, [-1, three_count])
    # Compute mean vector
    mean_v = np.sum(three_vec, 1)/three_count
    three_vec_sub_mean = (three_vec.transpose() - mean_v).transpose()
    [u, s, v] = np.linalg.svd(three_vec_sub_mean)

    pc2 = u[:, 0:2]
    S = np.diag(s)
    w2 = np.dot(S[:2], v)

    recovered_three_vec = (np.dot(pc2, w2).transpose() + mean_v).transpose()
    recovered_three = np.reshape(recovered_three_vec, [16, 16, three_count])

    grid_index = np.zeros([5, 5])
    choose_points = np.zeros([2, 25])
    for i in range(5):
        for j in range(5):
            v = [-0.010 + 0.005*i, -0.010 + 0.005*j]
            abs_v = abs(w2.transpose() - v)
            dis = np.sum(abs_v, 1)
            grid_index[i, j] = np.argmin(dis)
            choose_points[..., i+j*5] = w2[:, grid_index[i, j]]

    plt.figure(1)
    plt.plot(w2[0, :], w2[1, :], 'go', choose_points[0, :], choose_points[1, :], 'ro')
    plt.grid(True)
    plt.title('PCA')
    plt.xlabel('1st Principle Component')
    plt.ylabel('2nd Principle Component')
    pca_result = np.zeros([32*5, 32*5])

    for i in range(5):
        for j in range(5):
            pca_result[i*32:(i+1)*32, j*32:(j+1)*32] = three_origin[..., grid_index[i, j]]
    plt.figure(2)
    plt.title('Result')
    plt.imshow(pca_result)
    plt.show()



