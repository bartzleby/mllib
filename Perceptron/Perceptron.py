#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW3 Perceptron implementation
#

import numpy as np


def Perceptron(T, xs, y, r=1, mode='standard'):
  '''Run perceptron algorithm for T epopchs
  Return learned weight vector.

  Arguments:
    T -- number of epochs to run
    xs -- example data, prepended with ones
    y -- example labels

  Keyword Arguments:
    r -- learning rate (Default: 1)
    mode -- (Default: standard)
  '''
  m = np.shape(xs)[0]
  d = np.shape(xs)[1]

  w = np.zeros(np.shape(xs)[1])
  a = np.zeros(np.shape(xs)[1])
  for i in range(T):
#    np.random.shuffle(xs);

    for i in range(m):
      if y[i]*(w@xs[i,:]) <= 0:
        w = w + r*(xs[i,:]*y[i])
    
      a += w

  if mode == 'average':
    return a
  return w

def PredictVoted(ws, cs, example):
  '''Returns prediction
  '''
  sum = 0
  for i in range(len(cs)):
    sum += cs[i]*np.sign(ws[i]@example)

  return np.sign(sum)

def VotedPerceptron(T, xs, y, r=1):
  '''Run voted perceptron algorithm for T epopchs
  Return learned weight vectors and prediction counts.

  Arguments:
    T -- number of epochs to run
    xs -- example data, prepended with ones
    y -- example labels

  Keyword Arguments:
    r -- learning rate (Default: 1)
  '''
  m = np.shape(xs)[0]
  d = np.shape(xs)[1]
  ws = []
  cs = []
  w = np.zeros(np.shape(xs)[1])
  for i in range(T):
#    np.random.shuffle(xs);

    for i in range(m):
      if y[i]*(w@xs[1,:]) <= 0:
        w = w + r*(xs[i,:]*y[i])
        ws.append(w)
        cs.append(1)

      else:
        cs[-1] += 1

  return ws, cs


def main():
  return

if __name__ == '__main__':
    main()


