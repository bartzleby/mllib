#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# April, 2021
# 
# HW4 bank note data set
# test SVM learned weights
#

import numpy as np
import pickle

from data.bank_note.note import features, dtype, labels_to_pmone
from SVM import *

def main():

  with open('./pickle/ssgd_res02.pkl', 'rb') as file:
    res = pickle.load(file) 

  res_a = res[0]
  res_b = res[1]

  w_a = res_a[0]
  w_b = res_b[0]
  print(w_a)
  print(w_b)


  datrain = np.genfromtxt("../data/bank_note/train.csv", delimiter=',', dtype=dtype)
  xs_train = np.ones(np.shape(datrain), dtype=dtype)
  y_train = labels_to_pmone(datrain[:,-1])
  xs_train[:,1:] = datrain[:,0:-1]

  datest = np.genfromtxt("../data/bank_note/test.csv", delimiter=',', dtype=dtype)
  xs_test = np.ones(np.shape(datest), dtype=dtype)
  y_test = labels_to_pmone(datest[:,-1])
  xs_test[:,1:] = datest[:,0:-1]

  m = np.shape(xs_test)[0]
  d = np.shape(xs_test)[1]

  a_errors = 0
  b_errors = 0
  for xi in range(m):
    if np.sign(w_a @ xs_test[xi,:]) != y_test[xi]:
      a_errors += 1
    if np.sign(w_b @ xs_test[xi,:]) != y_test[xi]:
      b_errors += 1

  print("error rate schedule A: ", a_errors/(xi+1))
  print("error rate schedule B: ", b_errors/(xi+1))


  return


if __name__ == '__main__':
    main()


