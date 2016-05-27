'''
Homework 3
Goal:
1. implement MOG in 2D case:
2. E-M method
'''

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy
import math
# Step 1: Generate data

mean = [3, 4]
cov = [1, 0], [0, 0.5]
data1 = np.random.multivariate_normal(mean, cov, (10, 10))
x1 = np.reshape(data1[..., 0], data1[..., 0].size)
y1 = np.reshape(data1[..., 1], data1[..., 1].size)

mean = [7, 8]
cov = [0.5, 0], [0, 1]
data2 = np.random.multivariate_normal(mean, cov, (10, 10))
x2 = np.reshape(data2[..., 0], data2[..., 0].size)
y2 = np.reshape(data2[..., 1], data2[..., 1].size)

mean = [4, 8]
cov = [0.8, 0], [0, 0.5]
data3 = np.random.multivariate_normal(mean, cov, (10, 10))
x3 = np.reshape(data3[..., 0], data3[..., 0].size)
y3 = np.reshape(data3[..., 1], data3[..., 1].size)

mean = [7, 3]
cov = [0.5, 0], [0, 0.8]
data4 = np.random.multivariate_normal(mean, cov, (10, 10))
x4 = np.reshape(data4[..., 0], data4[..., 0].size)
y4 = np.reshape(data4[..., 1], data4[..., 1].size)

plt.plot(x1, y1, 'ro', x2, y2, 'b*', x3, y3, 'y^', x4, y4, 'w^')
plt.axis([0, 10, 0, 10])


# Step 2: E-M method
N = 400
data = np.zeros([N, 2])
data[0:100, ...] = np.reshape(data1, [100, 2])
data[100:200, ...] = np.reshape(data2, [100, 2])
data[200:300, ...] = np.reshape(data3, [100, 2])
data[300:400, ...] = np.reshape(data4, [100, 2])


class MOG:
    # Randomly choose 4 data points as mean of Gaussian Distributions
    mean1 = random.choice(data)
    mean2 = random.choice(data)
    mean3 = random.choice(data)
    mean4 = random.choice(data)

    # Assume all distribution have  covariance matrix
    sigma1 = np.array([[1, 0], [0, 1]])
    sigma2 = np.array([[1, 0], [0, 1]])
    sigma3 = np.array([[1, 0], [0, 1]])
    sigma4 = np.array([[1, 0], [0, 1]])


# Gaussian function for matrix
def gaussian(x, mean, sigma):
    d = mean.size
    mat = np.mat(sigma)
    p = np.exp(-np.sum(np.array(np.mat(x - mean) * mat.I)*(x - mean)/2, 1))/(math.pow(2*math.pi, d/2)*np.linalg.det(sigma))
    return p


# Gaussian function for single data point. This function is used for verify the correctness of function gaussian() above
def gaussian2(x, mean, sigma):
    d = mean.size
    mat = np.mat(sigma)
    p = np.exp(-(x - mean)* mat.I*(x-mean).transpose()/2)/(math.pow(2*math.pi, d/2)*np.linalg.det(sigma))
    return p


# EM method
def EM(iter, mog):
    for i in range(iter):
        # E-step: softly assign examples to mixture componets
        p1_v = gaussian(data, mog.mean1, mog.sigma1)
        p2_v = gaussian(data, mog.mean2, mog.sigma2)
        p3_v = gaussian(data, mog.mean3, mog.sigma3)
        p4_v = gaussian(data, mog.mean4, mog.sigma4)

        p = np.sum(p1_v + p2_v + p3_v + p4_v)
        distance = np.sum(((data - mog.mean1).transpose() * p1_v) ** 2)
        distance += np.sum(((data - mog.mean2).transpose() * p2_v) ** 2)
        distance += np.sum(((data - mog.mean3).transpose() * p3_v) ** 2)
        distance += np.sum(((data - mog.mean4).transpose() * p4_v) ** 2)

        print('p = {}, distance = {}, distance/p = {}'.format(p, distance, distance/p))

        # M-step: re-estimate the parameters based on the soft assignments
        n1 = np.sum(p1_v)
        n2 = np.sum(p2_v)
        n3 = np.sum(p3_v)
        n4 = np.sum(p4_v)

        n = n1 + n2 + n3 + n4
        p1 = n1/n
        p2 = n2/n
        p3 = n3/n
        p4 = n4/n

        mog.mean1 = np.sum(p1_v * data.transpose(), 1)/n1
        mog.mean2 = np.sum(p2_v * data.transpose(), 1)/n2
        mog.mean3 = np.sum(p3_v * data.transpose(), 1)/n3
        mog.mean4 = np.sum(p4_v * data.transpose(), 1)/n4

        mog.sigma1 = np.dot((p1_v * (data - mog.mean1).transpose()), data - mog.mean1) / n1
        mog.sigma2 = np.dot((p2_v * (data - mog.mean2).transpose()), data - mog.mean2) / n2
        mog.sigma3 = np.dot((p3_v * (data - mog.mean3).transpose()), data - mog.mean3) / n3
        mog.sigma4 = np.dot((p4_v * (data - mog.mean4).transpose()), data - mog.mean4) / n4




mog = MOG()
EM(20, mog)
print(mog.mean1, mog.mean2, mog.mean3, mog.mean4)
print(mog.sigma1, mog.sigma2, mog.sigma3, mog.sigma4)
plt.show()

