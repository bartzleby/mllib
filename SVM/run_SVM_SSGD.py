#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# April, 2021
# 
# HW4 bank note data set
# run SVM in primal domain with
# stochastic sub-gradient descent
#

import numpy as np
import pickle

from data.bank_note.note import features, dtype, labels_to_pmone
from SVM import *

def schedule_a(gamma_0, t, d):
  gamma_t = gamma_0/(1+t*gamma_0/d)
  return gamma_t

def schedule_b(gamma_0, t, d):
  gamma_t = gamma_0/(1 + t)
  return gamma_t


def main():

  dat = np.genfromtxt("../data/bank_note/train.csv", delimiter=',', dtype=dtype)

  xs = np.ones((np.shape(dat)[0], np.shape(dat)[1]+1), dtype=dtype)
  xs[:,-1] = labels_to_pmone(dat[:,-1])
  xs[:,1:-1] = dat[:,0:-1]

  CS = np.array([100, 500, 700])/873

  ws_a = []
  ws_b = []

  ovals_a = []
  ovals_b = []

  for C in CS:
    print(C)
    (wa, oa) = SVM_P_SSGD(100, xs, C, \
          gamma_0=1, schedule=schedule_a, retobj=True)
    ws_a.append(wa)
    ovals_a.append(oa)

    (wb, ob) = SVM_P_SSGD(100, xs, C, \
          gamma_0=1, schedule=schedule_b, retobj=True)
    ws_b.append(wb)
    ovals_b.append(ob)


  with open('./pickle/ssgd_res.pkl', 'wb') as file: 
    pickle.dump(((ws_a, ovals_a), (ws_b, ovals_b)), file)


  return


if __name__ == '__main__':
    main()


