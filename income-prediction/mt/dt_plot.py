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

  with open('./pickle/tree-errors.pkl', 'rb') as file:
    errors = pickle.load(file) 


  training_errors = errors[0]
  test_errors = errors[1]


  xs = np.linspace(1, len(test_errors['entropy']), len(test_errors['entropy']))

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)

  ax.plot(xs, test_errors['entropy'], label='test errors')
  ax.plot(xs, training_errors['entropy'], label='training errors')

  ax.set_title('Decision Tree Errors With Depth')
  ax.set_xlabel('Max Tree Depth')
  ax.set_ylabel('error rate')
  ax.legend()

  plt.show()

  return


if __name__ == '__main__':
    main()


