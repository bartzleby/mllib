#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# February, 2021
# 
# 
# 

import numpy as np

import DecisionTree as tree
from DecisionTree import DecisionTreeNode as dtn
from DecisionTreeUtils import *


def main():
  dtype = "|U4"
  D = np.genfromtxt("data/tennis.csv", dtype=dtype, skip_header=1)
  attributes = ["Outlook", "Temperature", "Humidity", "Wind"]
  attribute_dict = get_attr_dict(D, attributes)
  decisionTree = ID3(D, attribute_dict, labeled=True, dtype=dtype, gain_metric="majority_error")

  return decisionTree


if __name__ == "__main__":
  main()
