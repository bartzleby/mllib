#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 bonus:
# credit default dataset
#

import numpy as np
import pandas as pd
import pickle

import data.default.defaultData as dd
from DecisionTree import numeric2median
from EnsembleLearning import *


def main():
  dtype = dd.dtype
  attribute_dict = dd.attribute_dict

  df = pd.read_csv("../data/default/default-of-credit-card-clients.csv", header=1)
  S = df.to_numpy(dtype=dtype)

  S[:,-1] = labels_to_pmone_01(S[:,-1])

  np.random.shuffle(S)
  train = S[0:24000,1:]
  test = S[24000:-1,1:]

  fdata, attribute_dict = numeric2median([train, test], attribute_dict);

  train = fdata[0]
  test = fdata[1]

  H = AdaBoost(500, train[:,0:-1], attribute_dict, train[:,-1], dtype=dtype)
  with open('./pickle/def_credit_run_adaboost.pkl', 'wb') as file: 
    pickle.dump(attribute_dict, file)
    pickle.dump(train, file)
    pickle.dump(test, file)
    pickle.dump(H, file)


  trees = BaggedTrees(2000, 500, train[:,0:-1], attribute_dict, train[:,-1], dtype=dtype)
  forest = RandomForest(2000, 500, train[:,0:-1], attribute_dict, train[:,-1], dtype=dtype, NumRandAttr=5)

  with open('./pickle/def_credit_run.pkl', 'wb') as file: 
    pickle.dump(attribute_dict, file) 
    pickle.dump(train, file)
    pickle.dump(test, file)
    pickle.dump(H, file)
    pickle.dump(trees, file)
    pickle.dump(forest, file)


  return


if __name__ == '__main__':
    main()


