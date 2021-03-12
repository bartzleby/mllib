#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 bank data set
# learn a bagged tree
# hypothesis
#

import numpy as np
import pickle

import data.bank.bankData as bd
from DecisionTree import numeric2median
from EnsembleLearning import *


def main():
  dtype = bd.dtype
  attribute_dict = bd.attribute_dict
  S = np.genfromtxt("../data/bank/train.csv", delimiter=',', dtype=dtype)
  S, attribute_dict = numeric2median([S], attribute_dict); S = S[0];

  trees = BaggedTrees(500, 500, S[:,0:-1], attribute_dict, S[:,-1], labeled=False, dtype=dtype)

  with open('./pickle/bagged_trees.pkl', 'wb') as file: 
    # A new file will be created 
    pickle.dump(trees, file)

  return


if __name__ == '__main__':
    main()


