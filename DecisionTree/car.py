#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# February, 2021
# 
# HW1 car data set
# training tree
# 

import numpy as np
from DecisionTree import classify
from DecisionTreeUtils import *

def print_table(error_dict):
  '''
  '''
  print('tree depth & ', end='')
  print(' & '.join(format(i+1, "d")  for i in range(len(list(error_dict.values())[0]))), end = ' \\\\\n')
  print('\\hline')
  for k, v in error_dict.items():
    print(k, end=' & ')
    print(' & '.join(format(x, "1.3f") for x in v), end = ' \\\\\n')

  return

def main():
  dtype = "|U12"
  test = np.genfromtxt("data/car/test.csv", delimiter=',', dtype=dtype)
  training = np.genfromtxt("data/car/train.csv", delimiter=',', dtype=dtype)
  answers_test = test[:,-1]
  answers_training = training[:,-1]
  attribute_dict = {"buying": ['vhigh', 'high', 'med', 'low'], \
        "maint": ['vhigh', 'high', 'med', 'low'], \
        "doors": ['2', '3', '4', '5more'],        \
        "persons": ['2', '4', 'more'],            \
        "lug_boot": ['small', 'med', 'big'],      \
        "safety": ['low', 'med', 'high'] }

  errors_test = {"entropy": [], "majority_error": [], "Gini_index": []}
  errors_training = {"entropy": [], "majority_error": [], "Gini_index": []}

  for i in range(6):
   for metric in list(errors_test.keys()):
    dtroot = ID3(training, attribute_dict, labeled=True, dtype=dtype, gain_metric=metric, max_depth=i+1)

    error_count = 0
    predictions_test = np.empty(np.shape(answers_test), dtype=answers_test.dtype)
    for testi in range(np.shape(test)[0]):
      test_example = list(test[testi,0:-1])
      predictions_test[testi] = classify(dtroot, test_example, list(attribute_dict.keys()))
      if predictions_test[testi] != answers_test[testi]:
        error_count += 1

    erri = error_count/(testi+1)
    errors_test[metric].append(erri) 


    error_count = 0
    predictions_training = np.empty(np.shape(answers_training), dtype=answers_training.dtype)
    for trtesti in range(np.shape(training)[0]):
      trtest_example = list(training[trtesti,0:-1])
      predictions_training[trtesti] = classify(dtroot, trtest_example, list(attribute_dict.keys()))
      if predictions_training[trtesti] != answers_training[trtesti]:
        error_count += 1

    erri = error_count/(trtesti+1)
    errors_training[metric].append(erri) 


  print_table(errors_test)
  print_table(errors_training)

  return


if __name__ == "__main__":
  main()
