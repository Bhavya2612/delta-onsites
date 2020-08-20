# -*- coding: utf-8 -*-
"""multivariate numpy

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-vtGiKVeIsdsNTyUnfcxZwXNObTv_1KD
"""

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  #placeholders don't work in tf.versions>2
# [:,np.newaxis] is to transform one-dimensional data into two-dimensional data

Y =[]
x_data = np.random.randn(50,2)
for i in range(0,50):
    y = np.sin(x_data[i][0]+np.square(x_data[i][1]))
    Y.append(y)
epsilon= 0.5
Y = np.array(Y) + epsilon

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
x = np.ndarray(dtype = float, shape=[50,2])
y_ = np.ndarray(dtype = float, shape=[50,1])
W1 = np.random.randn(1,10)
b1 = np.zeros(shape=(1,10))
assert (W1.shape == (1,10))
assert (b1.shape == (1,10))
W2 = np.random.randn(10,1)
b2 = np.zeros(shape=(1,1))

def gradientDescent(x, y, theta1,theta2,b1,b2, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        de = np.tanh(np.dot(x, theta1)+b1)
        final = np.tanh(np.dot(de, theta2)+b2)

        loss = (final - y) ** 2
        cost = np.sum(loss ** 2) / (2 * m)
        #print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m                                                           # this is the wrong part in this. I was unable to compute the cost,hence
        # update                                                                                      #the gradient. The formula needs to be updated. this works for lin reg.
        theta2 = theta2 - alpha * gradient
    return theta2

m = len(x_data)
alpha = 0.1

pred = gradientDescent(x_data,Y_data,W1,W2,b1,b2,alpha,m,100)
pre = np.mean(pred)

prediction_y = np.multiply(x_data,pre)

plt.scatter(x_data,Y_data)
plt.plot(x_data,prediction_y)
plt.show()



