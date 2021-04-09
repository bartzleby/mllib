#!/usr/bin/env python3


import numpy as np
import income_info as ii
import pandas as pd

def main():
  attribute_dict = ii.attributes
  td = pd.read_csv("data/test_final.csv")

  td.drop(td.iloc[:, 1:], axis=1, inplace=True)
  td['Prediction']=0

  td.to_csv('dummy-submission.csv',index=False)

  return


if __name__ == "__main__":
  main()
