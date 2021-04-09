#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 concrete data set
# learn a LMS linear regression
# model with stochastic gradient descent
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

  r = 0.001
  costs = []
  iteration_threshold = int(1e5)
  # initialize weight vector to zero:
  w = np.zeros(d, dtype=dtype)
  costs.append(lms_cost_w(y, w, xs))
  for it in range(iteration_threshold):
    exi = np.random.randint(len(y))
    gradient = grad_LMS(y[exi], w, xs[exi,:], dtype=dtype)
    w = w - r*gradient
    costs.append(lms_cost_w(y, w, xs))


  with open('./pickle/conc_GDS.pkl', 'wb') as file: 
    pickle.dump(costs, file)
    pickle.dump(w, file)

  return


if __name__ == '__main__':
    main()


