#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 bank data set
# plot random
# forest errors

import matplotlib.pyplot as plt
import numpy as np
import pickle

def main():

  with open('./pickle/RandomForest_test_errors_bank_n.pkl', 'rb') as file:
    test_errors = pickle.load(file) 
  with open('./pickle/RandomForest_training_errors_bank_n.pkl', 'rb') as file:
    training_errors = pickle.load(file) 


  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)

  xs = np.linspace(1, len(test_errors[2]), len(test_errors[2]))

  for n in list(test_errors.keys()):
    ax.plot(xs, test_errors[n], label='test errors, n={}'.format(n))
    ax.plot(xs, training_errors[n], label='training errors, n={}'.format(n))


  ax.set_title('Random Forest Errors With Iterations')
  ax.set_xlabel('Iterations')
  ax.set_ylabel('Error Rate')
  ax.legend()

  plt.show()

  return


if __name__ == '__main__':
    main()


