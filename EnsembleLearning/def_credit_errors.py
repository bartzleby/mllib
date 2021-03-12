#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 credit derfault data set
# calculate errors

import matplotlib.pyplot as plt
import numpy as np
import pickle

from EnsembleLearning import *

def main():

  with open('./pickle/def_credit_trees.pkl', 'rb') as file:
    attribute_dict = pickle.load(file) 
    training = pickle.load(file)
    test = pickle.load(file)
    trees = pickle.load(file)
    forest = pickle.load(file)

  with open('./pickle/def_credit_run_100.pkl', 'rb') as file:
    attribute_dict = pickle.load(file) 
    training = pickle.load(file)
    test = pickle.load(file)
    H = pickle.load(file)
    trees_100 = pickle.load(file)
    forest_100 = pickle.load(file)

  attributes = list(attribute_dict.keys())

  print("Calculating AdaBoost test errors.")
  test_errors_adaboost = []
  for i in range(len(H[0])):
    error_count = 0
    predictions = []
    for testi in range(np.shape(test)[0]):
      test_example = list(test[testi,0:-1])
      test_prediction = classify_n_weak(i+1, H, test_example, attributes)
      if test_prediction != int(test[testi,-1]):
        error_count += 1

    test_errors_adaboost.append(error_count/(testi+1))

  print("Calculating AdaBoost trainingerrors.")
  training_errors_adaboost = []
  for i in range(len(H[0])):
    error_count = 0
    predictions = []
    for testi in range(np.shape(training)[0]):
      test_example = list(training[testi,0:-1])
      test_prediction = classify_n_weak(i+1, H, test_example, attributes)
      if test_prediction != int(training[testi,-1]):
        error_count += 1

    training_errors_adaboost.append(error_count/(testi+1))

  with open('./pickle/errors_def_credit_adaboost.pkl', 'wb') as file: 
    pickle.dump((training_errors_adaboost, test_errors_adaboost), file)


  print("Calculating bagging test errors.")
  test_errors_treebag = []
  for i in range(len(trees)):
    error_count = 0
    predictions = []
    for testi in range(np.shape(test)[0]):
      test_example = list(test[testi,0:-1])
      test_prediction = classify_from_tree_bag(trees[0:i+1], test_example, attributes)
      if test_prediction != test[testi,-1]:
        error_count += 1

    test_errors_treebag.append(error_count/(testi+1))

  print("Calculating bagging training errors.")
  training_errors_treebag = []
  for i in range(len(trees)):
    error_count = 0
    predictions = []
    for testi in range(np.shape(training)[0]):
      test_example = list(training[testi,0:-1])
      test_prediction = classify_from_tree_bag(trees[0:i+1], test_example, attributes)
      if test_prediction != training[testi,-1]:
        error_count += 1

    training_errors_treebag.append(error_count/(testi+1))

  with open('./pickle/errors_def_credit_treebag.pkl', 'wb') as file: 
    pickle.dump((training_errors_treebag, test_errors_treebag), file)


  print("Calculating randfor test errors.")
  test_errors_randfor = []
  for i in range(len(forest)):
    error_count = 0
    predictions = []
    for testi in range(np.shape(test)[0]):
      test_example = list(test[testi,0:-1])
      test_prediction = classify_from_tree_bag(forest[0:i+1], test_example, attributes)
      if test_prediction != test[testi,-1]:
        error_count += 1

    test_errors_randfor.append(error_count/(testi+1))

  print("Calculating AdaBoost errors.")
  training_errors_randfor = []
  for i in range(len(forest)):
    error_count = 0
    predictions = []
    for testi in range(np.shape(training)[0]):
      test_example = list(training[testi,0:-1])
      test_prediction = classify_from_tree_bag(forest[0:i+1], test_example, attributes)
      if test_prediction != training[testi,-1]:
        error_count += 1

    training_errors_randfor.append(error_count/(testi+1))

  with open('./pickle/errors_def_credit_randfor.pkl', 'wb') as file: 
    pickle.dump((training_errors_randfor, test_errors_ranbfor), file)


  return


if __name__ == '__main__':
    main()


