#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# February, 2021
# 
# HW1 car data set
# training tree
# 

import numpy as np
from DecisionTreeUtils import *

def classex(dtroot, example, attributes):
  '''Calissify input example from dtroot.
  Currently does not handle errors, e.g. 
  inputs not corresponding properly.

  params:
    dtroot -- root nod of decision tree ()
    example -- list of attribute values
          corrsponding to attribute input
    attributes -- list of attributes
  '''
  while dtroot.label is None:
    test_attr = dtroot.attribute
    tai = attributes.index(test_attr)
    dtroot = dtroot.classify(example[tai])
    del attributes[tai]
    del example[tai]

  return dtroot.label


def main():
  dtype = "|U12"
  D = np.genfromtxt("data/car/train.csv", delimiter=',', dtype=dtype)
  attribute_dict = {"buying": ['vhigh', 'high', 'med', 'low'], \
        "maint": ['vhigh', 'high', 'med', 'low'], \
        "doors": ['2', '3', '4', '5more'],        \
        "persons": ['2', '4', 'more'],            \
        "lug_boot": ['small', 'med', 'big'],      \
        "safety": ['low', 'med', 'high'] }

  

  dtroot = ID3(D, attribute_dict, labeled=True, dtype=dtype, gain_metric="entropy", max_depth=2)
  return dtroot


if __name__ == "__main__":
  main()
