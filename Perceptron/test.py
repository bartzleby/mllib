#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW3 bank note data set
#

import numpy as np
import pickle

from data.bank_note.note import features, dtype, labels_to_pmone
from Perceptron import *

def main():

  with open('./pickle/ws.pkl', 'rb') as file:
    ws = pickle.load(file) 

  w_std = ws[0]
  w_avg = ws[2]
  vot_res = ws[1]

#  print("w_std: ", w_std)
#  print("w_avg: ", w_avg)
  for i in range(len(vot_res[0])):
    print(vot_res[0][i])
  print("voted countss:\n", vot_res[1])

  dat = np.genfromtxt("../data/bank_note/test.csv", delimiter=',', dtype=dtype)
  xs = np.ones(np.shape(dat), dtype=dtype)
  y = labels_to_pmone(dat[:,-1])
  xs[:,1:] = dat[:,0:-1]

  m = np.shape(xs)[0]
  d = np.shape(xs)[1]

  std_errors = 0
  avg_errors = 0
  vot_errors = 0
  for xi in range(m):
    if np.sign(w_std @ xs[xi,:]) != y[xi]:
      std_errors += 1
    if np.sign(w_avg @ xs[xi,:]) != y[xi]:
      avg_errors += 1
    if PredictVoted(vot_res[0],vot_res[1], xs[xi,:]) != y[xi]:
      vot_errors += 1

#  print("error rate std Perceptron: ", std_errors/(xi+1))
#  print("error rate avg Perceptron: ", avg_errors/(xi+1))
#  print("error rate voted perceptron: ", vot_errors/(xi+1))


  return


if __name__ == '__main__':
    main()


