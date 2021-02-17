#!/usr/bin/env python3

# 
# CS5350 HW1, question 1
# Danny Bartz
# February, 2021
# 
# 

import numpy as np

import DecisionTree as tree
from DecisionTree import DecisionTreeNode as dtn
from DecisionTreeUtils import *

def main():
  dtype = np.int8
  D = np.genfromtxt("data/Tdata.csv", delimiter=",", dtype=dtype)
  attributes = ["x_1", "x_2", "x_3", "x_4"]
  attribute_dict = get_attr_dict(D, attributes)

  print("initial split gains:")
  for i, a in enumerate(attributes):
    print(a, Gain(D, i, labeled=True))
  print("We will split on x_2.")

  print(D)
  print("splits into")
  S = np.delete(D, [0,2,3], 0)
  S = np.delete(S, 1, 1)
  print(S)
  print("and")
  S = np.delete(D, [1,4,5,6], 0)
  S = np.delete(S, 1, 1)
  print(S)
  print("considering the latter")

  print("second split gains:")
  mod_attr = ["x_1", "x_3", "x_4"]
  for i, a in enumerate(mod_attr):
    print(a, Gain(S, i, labeled=True))
  print("We will split on x_4")
  print("to obtain two purely")
  print("labeled subsets.")

  S = np.delete(S, 2, 1)
  S0 = np.delete(S, 0, 0)
  S1 = np.delete(S, [1,2], 0)

  print(S0)
  print(S1)

  decisionTree = ID3(D, attribute_dict, labeled=True, dtype=dtype)

  return decisionTree


if __name__ == "__main__":
  main()
