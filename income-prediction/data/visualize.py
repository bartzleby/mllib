#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# April, 2021
# 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import income_info as ii
from DecisionTree import *

def main():
  #dtype = ii.dtype
  #attribute_dict = ii.attr_dict

  train = pd.read_csv("train_final.csv")
  test = pd.read_csv("test_final.csv")

  train_ages = train['age']
  train_genders = train['sex']

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  train_ages.value_counts(sort=False).sort_index(ascending=True).plot(kind='bar')
  plt.show()
  ax.set_title('Decision Tree Errors With Depth')
  ax.set_xlabel('Max Tree Depth')
  ax.set_ylabel('error rate')
  ax.legend()


  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  train_genders.value_counts().plot(kind='pie')
  plt.show()
  ax.set_title('Decision Tree Errors With Depth')
  ax.set_xlabel('Max Tree Depth')
  ax.set_ylabel('error rate')
  ax.legend()


  return



if __name__ == "__main__":
  main()
