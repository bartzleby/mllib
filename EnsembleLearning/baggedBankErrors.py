#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 bank data set
# calculate bagged
# trees errors
#

import numpy as np
import pickle
from statistics import mode

import data.bank.bankData as bd
from DecisionTree import numeric2median
from DecisionTree import classify
from EnsembleLearning import *


def main():
  dtype = bd.dtype
  attribute_dict = bd.attribute_dict
  attributes = list(attribute_dict.keys())

  # Open the file in binary mode 
  with open('./pickle/bagged_trees.pkl', 'rb') as file:
    trees = pickle.load(file)

  test = np.genfromtxt("../data/bank/test.csv", delimiter=',', dtype=dtype)
  training = np.genfromtxt("../data/bank/train.csv", delimiter=',', dtype=dtype)

  fdata, attribute_dict = numeric2median([training, test], attribute_dict)
  training = fdata[0]
  test = fdata[1]


  test_errors = []
  for i in range(len(trees)):
    error_count = 0
    predictions = []
    for testi in range(np.shape(test)[0]):
      test_example = list(test[testi,0:-1])
      test_prediction = classify_from_tree_bag(trees[0:i+1], test_example, attributes)
      if test_prediction != test[testi,-1]:
        error_count += 1

    test_errors.append(error_count/(testi+1))


  training_errors = []
  for i in range(len(trees)):
    error_count = 0
    predictions = []
    for testi in range(np.shape(training)[0]):
      test_example = list(training[testi,0:-1])
      test_prediction = classify_from_tree_bag(trees[0:i+1], test_example, attributes)
      if test_prediction != training[testi,-1]:
        error_count += 1

    training_errors.append(error_count/(testi+1))



  with open('./pickle/TreeBag_test_errors_bank.pkl', 'wb') as file: 
    # A new file will be created 
    pickle.dump(test_errors, file)
  with open('./pickle/TreeBag_training_errors_bank.pkl', 'wb') as file: 
    # A new file will be created 
    pickle.dump(training_errors, file)

  return


if __name__ == '__main__':
    main()


