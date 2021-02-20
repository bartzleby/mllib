#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# February, 2021
# 
# HW1 bank data set
# training tree
# 

import numpy as np
from DecisionTreeUtils import *

def classex(dtroot, example, attributes):
  '''Calissify input example from dtroot.
  Currently does not handle errors, e.g. 
  inputs not corresponding properly.

  params:
    dtroot -- root nod of decision tree ()
    example -- list of attribute values
          corrsponding to attribute input
    attributes -- list of attributes
  '''
  while dtroot.label is None:
    test_attr = dtroot.attribute
    tai = attributes.index(test_attr)
    dtroot = dtroot.classify(example[tai])
    del attributes[tai]
    del example[tai]

  return dtroot.label

def print_table(error_dict):
  '''
  '''
  print('tree depth & ', end='')

  print(' & '.join(format(s, "s") for s in list(error_dict.keys())), end = ' \\\\\n')
  print('\\hline')
  for i in range(len(list(error_dict.values())[0])):
    print(i+1, end=' & ')
    print(' & '.join(format(error_dict[x][i], '1.3f') for x in list(error_dict.keys())), end = ' \\\\\n')

  return

def main():
  dtype = "|U16"
  test = np.genfromtxt("data/bank/test.csv", delimiter=',', dtype=dtype)
  training = np.genfromtxt("data/bank/train.csv", delimiter=',', dtype=dtype)
  answers_test = test[:,-1]
  answers_training = training[:,-1]
  attribute_dict = {'age': 'numeric', \
        "job": ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",         \
                "blue-collar","self-employed","retired","technician","services"],                          \
    "marital": ["married","divorced","single"], "education": ["unknown","secondary","primary","tertiary"], \
    "default": ["yes", "no"], 'balance': 'numeric', "housing": ["yes", "no"], "loan": ["yes", "no"],       \
    "contact": ["unknown","telephone","cellular"], 'day': 'numeric',                                       \
      "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],       \
   'duration': 'numeric', 'campaign': 'numeric', 'pdays': 'numeric', 'previous': 'numeric',               \
   'poutcome': ["unknown","other","failure","success"] }

  # TODO: add fdata to loop for more compact code
  fdata, attribute_dict = numeric2median([training, test], attribute_dict)
  training = fdata[0]
  test = fdata[1]


  errors_test = {"entropy": [], "majority_error": [], "Gini_index": []}
  errors_training = {"entropy": [], "majority_error": [], "Gini_index": []}


  for i in range(16):
   for metric in list(errors_test.keys()):
    dtroot = ID3(training, attribute_dict, labeled=True, dtype=dtype, gain_metric=metric, max_depth=i+1)

    error_count = 0
    predictions_test = np.empty(np.shape(answers_test), dtype=answers_test.dtype)
    for testi in range(np.shape(test)[0]):
      test_example = list(test[testi,0:-1])
      predictions_test[testi] = classex(dtroot, test_example, list(attribute_dict.keys()))
      if predictions_test[testi] != answers_test[testi]:
        error_count += 1

    erri = error_count/(testi+1)
    errors_test[metric].append(erri) 


    error_count = 0
    predictions_training = np.empty(np.shape(answers_training), dtype=answers_training.dtype)
    for trtesti in range(np.shape(training)[0]):
      trtest_example = list(training[trtesti,0:-1])
      predictions_training[trtesti] = classex(dtroot, trtest_example, list(attribute_dict.keys()))
      if predictions_training[trtesti] != answers_training[trtesti]:
        error_count += 1

    erri = error_count/(trtesti+1)
    errors_training[metric].append(erri) 


  print_table(errors_test)
  print_table(errors_training)

  return dtroot


if __name__ == "__main__":
  main()
