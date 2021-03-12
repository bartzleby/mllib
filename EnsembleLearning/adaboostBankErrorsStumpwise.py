#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 bank data set
# calculate adaboost
# errors stumpwise
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


  test_errors_stumpwise = []
  training_errors_stumpwise = []

  for i in range(len(Hfinal[0])):
    test_error_count = 0
    training_error_count = 0

    for tsti in range(np.shape(test)[0]):
      tst_example = list(test[tsti,0:-1])
      tst_prediction = int(classify(Hfinal[1][i], tst_example, attributes))
      if tst_prediction != int(test[tsti,-1]):
        test_error_count += 1

    for trni in range(np.shape(training)[0]):
      trn_example = list(training[trni,0:-1])
      trn_prediction = int(classify(Hfinal[1][i], trn_example, attributes))
      if trn_prediction != int(training[trni,-1]):
        training_error_count += 1

    test_errors_stumpwise.append(test_error_count/(tsti+1))
    training_errors_stumpwise.append(training_error_count/(trni+1))


  with open('./pickle/AdaBoost_errors_stumpwise.pkl', 'wb') as file: 
    pickle.dump(test_errors_stumpwise, file)
    pickle.dump(training_errors_stumpwise, file)

  return


if __name__ == '__main__':
    main()


