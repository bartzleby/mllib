#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 Q1.5: test data set
#

import numpy as np
import pandas as pd

dtype = np.int8
from Regression import *


def main():

  data = np.genfromtxt("./data/test-data.csv", delimiter=',', dtype=dtype)
  df = pd.read_csv("./data/test-data.csv", header=None, names=['x_1', 'x_2', 'x_3', 'out'])

  xs = np.ones(np.shape(data), dtype=dtype)
  y = data[:,-1]
  xs[:,1:] = data[:,0:-1]

  m = np.shape(xs)[0]
  d = np.shape(xs)[1]

  print("gradient: ", grad_LMS_batch(y, [-1,-1,1,-1], xs, dtype=dtype))
  print("w_star: ", LMS_Analytic(xs, y))
  print()

  r = 0.1
  w = np.zeros(d, dtype=dtype)
  print("Performing a sequential stochastic Gradient descent.")
  print("r= ", r)
  print()
  for i in range(5):
    idx = i # np.random.randint(len(y))
    print("iteration: ", i)
    print("w: ", w)
    print("cost: ", lms_cost_w(y, w, xs))
    gradient = grad_LMS(y[idx], w, xs[idx,:], dtype=dtype)
    print("gradient: ", gradient)
    wit = w - r*gradient
    w = wit

    print("new w: ", w)
    print()



#  r = 0.05
#  w = np.zeros(d, dtype=dtype)
#  print("Performing batch Gradient descent.")
#  print("r= ", r)
#  for i in range(50):
#    print("iteration: ", i)
#    print("w: ", w)
#    gradient = grad_LMS_batch(y, w, xs)
#    print("gradient: ", gradient)
#    wit = w - r*gradient
#    w = wit
#
#    print("new w: ", w)
#    print()


  return


if __name__ == '__main__':
    main()


