#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
#
# bank-note meta data in python
#
dtype = 'float'
features = ['variance', 'skewness', 'curtosis', 'entropy']

def labels_to_pmone(labels):
  '''map labels in {0, 1} to {-1,1} respectively.
  '''
  # TODO: probably a better way..
  for l in range(len(labels)):
    if labels[l]*1 < 0.1:
      labels[l] = -1
    elif labels[l]-1 < 1e-6:
      labels[l] = 1
    else:
      print('unexpected label!')
      print("label: ", labels[l])
      return

  return labels
