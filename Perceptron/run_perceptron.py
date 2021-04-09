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

  dat = np.genfromtxt("../data/bank_note/train.csv", delimiter=',', dtype=dtype)

  xs = np.ones(np.shape(dat), dtype=dtype)
  y = labels_to_pmone(dat[:,-1])
  xs[:,1:] = dat[:,0:-1]

  w_std = Perceptron(10, xs, y, r=1, mode='standard')
  ws_vot, cs_vot = VotedPerceptron(10, xs, y, r=1)
  w_avg = Perceptron(10, xs, y, r=1, mode='average')

  with open('./pickle/ws.pkl', 'wb') as file: 
    pickle.dump((w_std, (ws_vot, cs_vot), w_avg), file)


  return


if __name__ == '__main__':
    main()


