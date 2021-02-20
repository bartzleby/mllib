#!/usr/bin/env python3


import numpy as np
import income_info as ii

from DecisionTreeUtils import *

def main():
  ftype = "|U12"
  attribute_dict = ii.attr_dict
  print(attr_dict)
  

  td.drop(td.iloc[:, 1:], axis=1, inplace=True)
  td['Prediction']=0

  td.to_csv('dummy-submission.csv',index=False)

  return


if __name__ == "__main__":
  main()
