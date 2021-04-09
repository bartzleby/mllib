#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# income level prediction
# with decision tree
# 

import numpy as np
import pickle

import income_info as ii
from DecisionTree import *

def main():
  dtype = ii.dtype
  attribute_dict = ii.attr_dict

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

  dtroot = ID3(train, attribute_dict, labeled=True, dtype=dtype, gain_metric='entropy', max_depth=3)
  for testi in range(np.shape(test)[0]):
    predictions[testi+1,1] = classify(dtroot, list(test[testi,:]), list(attribute_dict.keys()))

  np.savetxt('dt_pred.csv', predictions, delimiter=',', fmt='%s')

  return


if __name__ == "__main__":
  main()
