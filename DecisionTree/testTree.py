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
  decisionTree = ID3(D, attribute_dict, labeled=True, dtype=dtype)

  return decisionTree


if __name__ == "__main__":
  main()
