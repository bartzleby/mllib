#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 bank data set
# learn a random forest
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

  forests = {}
  for n in [2,4,6]:
    forest = RandomForest(500, 500, S[:,0:-1], attribute_dict, S[:,-1], labeled=False, dtype=dtype, NumRandAttr=n)
    forests.update({n: forest})

  with open('./pickle/random_forests.pkl', 'wb') as file: 
    # A new file will be created 
    pickle.dump(forests, file)

  return forests


if __name__ == '__main__':
    main()


