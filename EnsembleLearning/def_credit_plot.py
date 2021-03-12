#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 bank data set
# plot adaboost
# errors

import matplotlib.pyplot as plt
import numpy as np
import pickle

def main():

  with open('./pickle/errors_def_credit_adaboost.pkl', 'rb') as file:
    errors = pickle.load(file)

  training_errors_adaboost = errors[0]
  test_errors_adaboost = errors[1]


  xs = np.linspace(1, len(test_errors_adaboost), len(test_errors_adaboost))

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)

  ax.plot(xs, test_errors_adaboost, label='AdaBoost test errors')
  ax.plot(xs, training_errors_adaboost, label='AdaBoosttraining errors')

  ax.set_title('Errors With Iterations')
  ax.set_xlabel('iterations')
  ax.set_ylabel('error rate')
  ax.legend()

  plt.show()

  return


if __name__ == '__main__':
    main()


