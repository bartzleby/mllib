#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# February, 2021
# 
# 
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
  dtype = "|U4"
  D = np.genfromtxt("data/tennis.csv", dtype=dtype, skip_header=1)
  attributes = ["Outlook", "Temperature", "Humidity", "Wind"]

  # hacky, but we don't care:
  attribute_dict = get_attr_dict(D, attributes)
  attribute_dict["Humidity"].append('L')
  # attr_dict will generally come from a data description

  dtroot = ID3(D, attribute_dict, labeled=True, gain_metric="Gini_index", dtype=dtype, display=True)

  return dtroot


if __name__ == "__main__":
  main()
