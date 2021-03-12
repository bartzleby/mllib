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

  with open('./pickle/def_credit_run_100.pkl', 'rb') as file:
    attribute_dict = pickle.load(file) 
    train = pickle.load(file)
    test = pickle.load(file)

  print('learning tree bags')
  trees = BaggedTrees(2000, 500, train[:,0:-1], attribute_dict, train[:,-1], dtype=dtype)
  print('learning random forest')
  forest = RandomForest(2000, 500, train[:,0:-1], attribute_dict, train[:,-1], dtype=dtype, NumRandAttr=5)

  with open('./pickle/def_credit_trees.pkl', 'wb') as file:
    pickle.dump(attribute_dict, file)
    pickle.dump(train, file)
    pickle.dump(test, file)
    pickle.dump(trees, file)
    pickle.dump(forest, file)


  return


if __name__ == '__main__':
    main()


