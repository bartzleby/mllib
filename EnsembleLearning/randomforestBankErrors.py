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


def classify_from_tree_bag(trees, example, attributes):
  '''Classify example using a bag (list) of decision trees.

  Returns predicted label

  Arguments:
    trees -- list of learned decision tree roots
    example -- example to be classified
    attributes -- list of attributes
  '''
  preds = {}
  for root in trees:
    pred = classify(root, example, attributes)
    if pred in preds:
      preds[pred] += 1
    else:
      preds.update({pred: 1})
  
  majority_vote = max(preds, key=preds.get)
  return majority_vote


def main():
  dtype = bd.dtype

  # Open the file in binary mode 
  with open('./pickle/random_forests.pkl', 'rb') as file:
    forests = pickle.load(file)


  test_errors = {}
  training_errors = {}

  for n in [2, 4, 6]:

    print("n: ", n)
    trees = forests[n]

    test = np.genfromtxt("../data/bank/test.csv", delimiter=',', dtype=dtype)
    training = np.genfromtxt("../data/bank/train.csv", delimiter=',', dtype=dtype)

    attribute_dict = dict(bd.attribute_dict)
    attributes = list(attribute_dict.keys())
    fdata, attribute_dict = numeric2median([training, test], attribute_dict)
    training = fdata[0]
    test = fdata[1]


    test_errors.update({n: []})
    for i in range(len(trees)):
      error_count = 0
      predictions = []
      for testi in range(np.shape(test)[0]):
        test_example = list(test[testi,0:-1])
        test_prediction = classify_from_tree_bag(trees[0:i+1], test_example, attributes)
        if test_prediction != test[testi,-1]:
          error_count += 1

      test_errors[n].append(error_count/(testi+1))


    training_errors.update({n: []})
    for i in range(len(trees)):
      error_count = 0
      predictions = []
      for testi in range(np.shape(training)[0]):
        test_example = list(training[testi,0:-1])
        test_prediction = classify_from_tree_bag(trees[0:i+1], test_example, attributes)
        if test_prediction != training[testi,-1]:
          error_count += 1

      training_errors[n].append(error_count/(testi+1))



  with open('./pickle/RandomForest_test_errors_bank_n.pkl', 'wb') as file: 
    # A new file will be created 
    pickle.dump(test_errors, file)
  with open('./pickle/RandomForest_training_errors_bank_n.pkl', 'wb') as file: 
    # A new file will be created 
    pickle.dump(training_errors, file)

  return


if __name__ == '__main__':
    main()


