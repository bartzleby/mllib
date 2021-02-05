#!/usr/bin/env python3

# 
# CS5350 HW1, question 1
# Danny Bartz
# February, 2021
# 
# 
# 

import numpy as np


def load_data(rel_path):
  # returns anarray:
  data = np.genfromtxt(rel_path, delimiter=",")
  y = data[:,-1]
  ex = data[:,0:-1]
  return data


#
# Function to return the entropy of
# a set S of examples, including
# labels as rightmost column.
# 
# Can also accept a one dimensional
# array of labels as input S, as input
#
# returns -1 if dimension > 2
def Entropy(S):
  if S.ndim > 2:
    return -1
  if S.ndim == 2:
    S = S[:,-1]

  #create a dict to track possible labels and counts:
  label_dict = {}
  for k in S:
    if k in label_dict:
      label_dict[k] += 1
    else:
      label_dict[k] = 1

  nex = len(S) # total examples in set
  H = 0;       # Entropy
  for label, count in label_dict.items():
    H -= (count/nex)*np.log(count/nex)

  return H

#
# Function to return the information gain of 
# partitioning set S on attribute A, an index
# such that S[i, A] is Value(A) for example i.
# 
# rows in S are examples, with S[i, -1]
# being the label for example i
#
def Gain(S, A):
  nex = S.shape[0] # number of examples
  HS = Entropy(S)

  # we collect all values A takes in S, 
  # and track row indices:
  values_dict = {}
  for i in range(nex):
    if S[i, A] in values_dict:
      values_dict[S[i, A]].append(i)
    else:
      values_dict[S[i, A]] = [i]

  G = HS
  for value, indices in values_dict.items():
    Sv = np.empty((len(indices), S.shape[1]))
    for i, r in enumerate(indices):
      Sv[i,:] = S[r,:]

    HSv = Entropy(Sv)
    G -= (len(indices)/nex)*HSv

  return G


def main():
  D = load_data("data/Tdata.csv")
  
  for i in range(D.shape[1]-1):
    print(i, Gain(D, i))

  return


if __name__ == "__main__":
  main()