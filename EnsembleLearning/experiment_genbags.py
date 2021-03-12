#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 bank data set
# learn a hundred bagged
# tree hypotheses
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

  bags = []
  for i in range(100):
    print("generating bag ", i)
    np.random.shuffle(S)
    bags.append(BaggedTrees(500, 500, S[0:1000,0:-1], attribute_dict, S[0:1000,-1], dtype=dtype))

  with open('./pickle/experiment_bags_m500.pkl', 'wb') as file: 
    # A new file will be created 
    pickle.dump(bags, file)

  return


if __name__ == '__main__':
    main()


