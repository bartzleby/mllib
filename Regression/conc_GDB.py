#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 concrete data set
# learn a LMS linear regression
# model with batch gradient descent
#

import numpy as np
import pandas as pd
import pickle

from data.concrete.concreteSLUMP import features, dtype
from Regression import *


def main():

  data = np.genfromtxt("../data/concrete/train.csv", delimiter=',', dtype=dtype)
  df = pd.read_csv("../data/concrete/train.csv", header=None, names=features)

  xs = np.ones(np.shape(data), dtype=dtype)
  y = data[:,-1]
  xs[:,1:] = data[:,0:-1]

  m = np.shape(xs)[0]
  d = np.shape(xs)[1]

  r = 0.015
  convergence_threshold = 1e-6
  iteration_threshold   = 1e7

  costs = []
  # initialize weight vector to zero:
  w = np.zeros(d, dtype=dtype)
  costs.append(lms_cost_w(y, w, xs))
  dw_norm = np.float('inf')
  it = 0
  while dw_norm > convergence_threshold and it < iteration_threshold:
    gradient = grad_LMS_batch(y, w, xs)
    wit = w - r*gradient
    costs.append(lms_cost_w(y, wit, xs))
    it += 1
    dw_norm = np.linalg.norm(wit-w)
    w = wit


  with open('./pickle/conc_GDB.pkl', 'wb') as file: 
    pickle.dump(costs, file)
    pickle.dump(w, file)


  return w


if __name__ == '__main__':
    main()


