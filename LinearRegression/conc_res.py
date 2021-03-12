#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 concrete data set
# evaluate errors of a LMS linear
# regression model learned with 
# gradient descent
#

import matplotlib.pyplot as plt
import numpy as np
import pickle

from data.concrete.concreteSLUMP import features, dtype
from Regression import *

def main():

  with open('./pickle/conc_GDB.pkl', 'rb') as file: 
    batch_costs = pickle.load(file)
    w_batch = pickle.load(file)

  with open('./pickle/conc_GDS.pkl', 'rb') as file: 
    stochastic_costs = pickle.load(file)
    w_stochastic = pickle.load(file)


  xaxs_batch = np.linspace(0, len(batch_costs)-1, len(batch_costs))
  xaxs_stochastic = np.linspace(0, len(stochastic_costs)-1, len(stochastic_costs))

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)

  ax.plot(xaxs_batch, batch_costs, label='batch GD costs, r=0.015')
  ax.plot(xaxs_stochastic, stochastic_costs, label='stochastic GD costs, r=0.001')

  ax.set_title('Gradient Descent Cost With Iteration')
  ax.set_xlabel('iterations')
  ax.set_ylabel('LMS cost')
  ax.legend()

#  plt.show()




  test = np.genfromtxt("../data/concrete/test.csv", delimiter=',', dtype=dtype)
  xs = np.ones(np.shape(test), dtype=dtype)
  y = test[:,-1]
  xs[:,1:] = test[:,0:-1]

  print("w_batch: ", w_batch)
  print("test data cost for w_batch: ", lms_cost_w(y, w_batch, xs))
  print("w_stochastic: ", w_stochastic)
  print("test data cost for w_stochastic: ", lms_cost_w(y, w_stochastic, xs))


  train = np.genfromtxt("../data/concrete/train.csv", delimiter=',', dtype=dtype)
  xs = np.ones(np.shape(train), dtype=dtype)
  y = train[:,-1]
  xs[:,1:] = train[:,0:-1]

  w_opt = LMS_Analytic(xs, y)
  print("w_opt: ", w_opt)
  print("training data cost for w_opt: ", lms_cost_w(y, w_opt, xs))
  print("training data cost for w_batch: ", lms_cost_w(y, w_batch, xs))
  print("training data cost for w_stochastic: ", lms_cost_w(y, w_stochastic, xs))


  return


if __name__ == '__main__':
    main()


