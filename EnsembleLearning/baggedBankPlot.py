#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 bank data set
# plot bagged
# trees errors

import matplotlib.pyplot as plt
import numpy as np
import pickle

def main():

  with open('./pickle/TreeBag_test_errors_bank.pkl', 'rb') as file:
    test_errors = pickle.load(file) 
  with open('./pickle/TreeBag_training_errors_bank.pkl', 'rb') as file:
    training_errors = pickle.load(file) 

  xs = np.linspace(1, len(test_errors), len(test_errors))

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)

  ax.plot(xs, test_errors, label='test errors')
  ax.plot(xs, training_errors, label='training errors')

  ax.set_title('Bagged Trees Errors With Iterations')
  ax.set_xlabel('Bagging Iiterations')
  ax.set_ylabel('Error Rate')
  ax.legend()

  plt.show()

  return


if __name__ == '__main__':
    main()


