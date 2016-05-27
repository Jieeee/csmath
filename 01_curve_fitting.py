# implement polynomial curve fitting in python

# step 1: prepare data
import numpy as np
import matplotlib.pyplot as plt

# generate sin(x) data with x in [0, 6)
X = np.arange(0, 6, 0.006)
y = np.sin(X)

# generate (X, y) data with 10 samples
X_10 = np.arange(0, 6, 0.6)
y_10 = np.sin(X_10)

# add norm distribution noise to samples
y_noise_10 = y_10 + 0.1*np.random.randn(10)


def rms(p, t):
    #size = len(p)
    loss = (p-t)**2/2
    # rms_loss = np.sqrt(loss)
    return loss


def model_output(model, x):
    index = 0
    degree = len(model)
    output = np.zeros(degree)
    while index < len(model):
        output += model[index] * (x ** index)
        index += 1

    return np.sum(output)


def model_grad(model, a_x, a_y):
    grad = np.zeros(len(model))
    index = 0
    while index < len(model):
        grad[index] = (model_output(model, a_x)-a_y)*(a_x**index)
        index += 1

    return grad


def sgd(model, train_x, train_y, lr=0.001, it=10):
    # degree = len(model)
    print(model)
    for i in np.arange(it):
        loss = 0.0
        index = 0
        while index < len(train_x):
            grad = model_grad(model, train_x[index], train_y[index])
            # print('grad:', grad)
            output = model_output(model, train_x[index])
            loss += rms(output, train_y[index])
            model -= lr*grad
            index += 1

        print('[INFO] Iteration %d, Loss = %f' % (i, loss))


# fit degree 3 and 9 in 10 samples
model_3 = np.random.randn(4)
model_9 = np.random.randn(10)

# No training
p_y = np.zeros(1000)
index = 0
while index < 1000:
    p_y[index] = model_output(model_3,X[index])
    index += 1
plt.figure(1)
plt.suptitle('Training result')
plt.subplot(221)
plt.plot(X, y, 'b-', X_10, y_10, 'ro', X, p_y, 'r-')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
#plt.axis([0, 6, -1, 1])
plt.title('No training')
# plt.show()

# train model with 10000 iterations
sgd(model_3, X_10, y_10, 0.00001, 10000)
index = 0
while index < 1000:
    p_y[index] = model_output(model_3,X[index])
    index += 1
plt.subplot(222)
plt.plot(X, y, 'b-', X_10, y_10, 'ro', X, p_y, 'r-')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
#plt.axis([0, 6, -1, 1])
plt.title('10000 iterations')

# train model with 50000 iterations
sgd(model_3, X_10, y_10, 0.00001, 40000)
index = 0
while index < 1000:
    p_y[index] = model_output(model_3,X[index])
    index += 1
plt.subplot(223)
plt.plot(X, y, 'b-', X_10, y_10, 'ro', X, p_y, 'r-')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
#plt.axis([0, 6, -1, 1])
plt.title('50000 iterations')

# train model with 100000 iterations
sgd(model_3, X_10, y_10, 0.00001, 50000)
index = 0
while index < 1000:
    p_y[index] = model_output(model_3,X[index])
    index += 1
plt.subplot(224)
plt.plot(X, y, 'b-', X_10, y_10, 'ro', X, p_y, 'r-')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
#plt.axis([0, 6, -1, 1])
plt.title('100000 iterations')
plt.show()
