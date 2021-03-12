#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 bank data set
# calculate adaboost
# errors
#

import numpy as np
import pickle

import data.bank.bankData as bd
from DecisionTree import numeric2median
from DecisionTree import classify
from EnsembleLearning import *


def main():
  dtype = bd.dtype
  attribute_dict = bd.attribute_dict
  attributes = list(attribute_dict.keys())

  # Open the file in binary mode 
  with open('./pickle/Hfinal.pkl', 'rb') as file:
    Hfinal = pickle.load(file) 

  test = np.genfromtxt("../data/bank/test.csv", delimiter=',', dtype=dtype)
  training = np.genfromtxt("../data/bank/train.csv", delimiter=',', dtype=dtype)

  fdata, attribute_dict = numeric2median([training, test], attribute_dict)
  training = fdata[0]
  test = fdata[1]

  training[:,-1] = labels_to_pmone(training[:,-1]);  
  test[:,-1] = labels_to_pmone(test[:,-1]);  


  test_errors = []
  for i in range(len(Hfinal[0])):
    error_count = 0
    predictions = []
    for testi in range(np.shape(test)[0]):
      test_example = list(test[testi,0:-1])
      test_prediction = classify_n_weak(i+1, Hfinal, test_example, attributes)
      if test_prediction != int(test[testi,-1]):
        error_count += 1

    test_errors.append(error_count/(testi+1))


  training_errors = []
  for i in range(len(Hfinal[0])):
    error_count = 0
    predictions = []
    for testi in range(np.shape(training)[0]):
      test_example = list(training[testi,0:-1])
      test_prediction = classify_n_weak(i+1, Hfinal, test_example, attributes)
      if test_prediction != int(training[testi,-1]):
        error_count += 1

    training_errors.append(error_count/(testi+1))





  with open('./pickle/AdaBoost_test_errors_bank.pkl', 'wb') as file: 
    # A new file will be created 
    pickle.dump(test_errors, file)
  with open('./pickle/AdaBoost_training_errors_bank.pkl', 'wb') as file: 
    # A new file will be created 
    pickle.dump(training_errors, file)

  return


if __name__ == '__main__':
    main()


