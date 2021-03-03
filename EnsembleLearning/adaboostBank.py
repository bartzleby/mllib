#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 bank data set
# adaboost
#

import numpy as np

import data.bank.bankData as bd
from DecisionTree import numeric2median
from EnsembleLearning import *


def main():
  dtype = bd.dtype
  attribute_dict = bd.attribute_dict

  test = np.genfromtxt("../data/bank/test.csv", delimiter=',', dtype=dtype)
  training = np.genfromtxt("../data/bank/train.csv", delimiter=',', dtype=dtype)

  fdata, attribute_dict = numeric2median([training, test], attribute_dict)
  training = fdata[0]
  test = fdata[1]
  
  # we need to map labels {no, yes} to {-1, 1} respectively
  # TODO: probably a better way..
  for r in range(np.shape(training)[0]):
    if training[r,-1] == 'no':
      training[r,-1] = -1
    elif training[r,-1] == 'yes':
      training[r,-1] = 1
    else:
      print(training[r,:])
      print('unexpected label!')
      return 0

  H = AdaBoost(10, training[:,0:-1], attribute_dict, labels=training[:,-1], labeled=False, dtype=dtype)


  return


if __name__ == '__main__':
    main()


