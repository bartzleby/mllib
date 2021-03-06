#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 concrete data set
# learn a linear regression
# model with batch 
# gradient descent
#

import numpy as np
import pandas as pd
import pickle

from data.concrete.concreteSLUMP import features, dtype


def main():
  model = []
  S = np.genfromtxt("../data/concrete/train.csv", delimiter=',', dtype=dtype)

  with open('./pickle/concrete_reg_batchGD.pkl', 'wb') as file: 
    pickle.dump(model, file)

  return model


if __name__ == '__main__':
    main()


