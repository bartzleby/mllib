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
  D = np.genfromtxt("data/tennis-missing.csv", dtype=dtype, skip_header=1, missing_values="?")
  attributes = ["Outlook", "Temperature", "Humidity", "Wind"]


  attribute_dict = get_attr_dict(D, attributes)
  attribute_dict["Humidity"].append('L')

  labels = Labels(D[:,-1])
  D = np.delete(D, -1, axis=1)
  D = labels.assign_fractional_counts(D)


  decisionTree = ID3(D, attribute_dict, labels=labels, dtype=dtype, display=True)


  return decisionTree


if __name__ == "__main__":
  main()
