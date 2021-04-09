#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# predict from 
# adaboost
#

import numpy as np
import pickle

import income_info as ii
from DecisionTree import *
from EnsembleLearning import *

def main():
  dtype = ii.dtype
  attribute_dict = ii.attr_dict
  attributes = list(attribute_dict.keys())

  train = np.genfromtxt("data/train_final.csv", delimiter=',', dtype=dtype, skip_header=1)
  test = np.genfromtxt("data/test_final.csv", delimiter=',', dtype=dtype, skip_header=1)

  predictions = np.empty((np.shape(test)[0]+1, 2), dtype=dtype)
  predictions[0,:] = ['ID','Prediction']
  predictions[1:,0] = test[:,0]

  np.random.shuffle(train)
  fdata, attribute_dict = numeric2median([train, test[:,1:]], attribute_dict, indicator="continuous")
  train = fdata[0]
  test = fdata[1]

  train, mcvs = assign_most_common_general(train, missing_values='?')
  test = assign_most_common_general(test, missing_values='?', mcvs=mcvs)[0]

  train[:,-1] = labels_to_pmone_01(train[:,-1])

  with open('./pickle/Hfinal.pkl', 'rb') as file:
    Hfinal = pickle.load(file) 



  training_errors = []
  for i in range(len(Hfinal[0])):
    print("training errors ", i)
    error_count = 0
    for testi in range(np.shape(train)[0]):
      test_example = list(train[testi,0:-1])
      test_prediction = classify_n_weak(i+1, Hfinal, test_example, attributes)
      if test_prediction != int(train[testi,-1]):
        error_count += 1

    training_errors.append(error_count/(testi+1))

  print("test errors ", np.shape(Hfinal)[1])
  for testi in range(np.shape(test)[0]):
    test_example = list(test[testi,:])
    test_prediction = classify_n_weak(np.shape(Hfinal)[1], Hfinal, test_example, attributes)
    if (test_prediction < 1):
      predictions[testi+1,1] = '0'
    else:
      predictions[testi+1,1] = '1'

  np.savetxt('ada_pred.csv', predictions, delimiter=',', fmt='%s')
  with open('./pickle/ada_pred.pkl', 'wb') as file:
    pickle.dump(training_errors, file)
    pickle.dump(predictions, file)

  return


if __name__ == '__main__':
    main()
