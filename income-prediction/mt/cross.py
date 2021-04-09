#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# income level prediction
# learn decision trees
# to determine best
# tree depth
# 

import numpy as np
import pickle

import income_info as ii
from DecisionTree import *

def main():
  dtype = ii.dtype
  attribute_dict = ii.attr_dict

  data = np.genfromtxt("data/train_final.csv", delimiter=',', dtype=dtype, skip_header=1)
  np.random.shuffle(data)
  fdata, attribute_dict = numeric2median([data[:15000,:], data[15000:,:]], attribute_dict, indicator="continuous")
  train = fdata[0]
  test = fdata[1]

  train, mcvs = assign_most_common_general(train, missing_values='?')
  test = assign_most_common_general(test, missing_values='?', mcvs=mcvs)[0]

  answers_test = test[:,-1]
  answers_training = train[:,-1]

  errors_test = {"entropy": [], "majority_error": [], "Gini_index": []}
  errors_training = {"entropy": [], "majority_error": [], "Gini_index": []}


  for i in range(14):
   for metric in list(errors_test.keys()):
    print(i, metric)
    dtroot = ID3(train, attribute_dict, labeled=True, dtype=dtype, gain_metric=metric, max_depth=i+1)

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
    for trtesti in range(np.shape(train)[0]):
      trtest_example = list(train[trtesti,0:-1])
      predictions_training[trtesti] = classify(dtroot, trtest_example, list(attribute_dict.keys()))
      if predictions_training[trtesti] != answers_training[trtesti]:
        error_count += 1

    erri = error_count/(trtesti+1)
    errors_training[metric].append(erri) 



  with open('./pickle/tree-errors.pkl', 'wb') as file: 
    pickle.dump((errors_training, errors_test), file)


  return


if __name__ == "__main__":
  main()
