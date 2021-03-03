#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# Ensemble learning routines
# 



import numpy as np

import DecisionTree as dt

default_dtype = dt.default_dtype


def weighted_error(stump, S, attributes, labels=None, labeled=False, weights=None):
  '''Return weighted error of given decision stump.
  Classified with classify function from DecisionTree.
  Labels must be in {-1, 1}

  Arguments:
  stump -- DecisionTree stump node
  S -- example data, no header
  attributes -- list of attributes
  Dt -- example weights

  Keyword arguments:
  labels -- list of labels (Default: None)
  labeled -- is data labeled? if so must set to True (Default: False)
  '''
  labels = dt.check_labels(S, labels=labels, labeled=labeled)
  if weights is None:
    Dt = labels.fractional_counts
  else:
    Dt = weights

  et = 1/2
  for i in range(len(Dt)):
    et -= 0.5*Dt[i]*float(labels[i])*float(dt.classify(stump, list(S[i,:]), list(attributes)))

  return et


def create_stump_prime(weights, S, attribute_dict, labels=None, labeled=False, dtype=default_dtype):
  '''Returns a decision stump to minimize resulting weighted error.
  '''
  labels = dt.check_labels(S, labels=labels, labeled=labeled)
  attributes = list(attribute_dict.keys())

  minerr = 1
  stump = None
  for a, attr in enumerate(attributes):
    test_stump = dt.DecisionTreeNode(attr, attribute_dict[attr])
    for attr_val in test_stump.values:
      Sv, Sv_labels = dt.get_Sv(S, a, attr_val, labels, dtype=dtype)
      test_stump.branches.update({attr_val: dt.DecisionTreeNode(label=Sv_labels.most_common())})

    we = weighted_error(test_stump, S, attributes, weights, labels=labels, labeled=labeled)
    if we < minerr:
      minerr = we
      stump = test_stump


  return stump


def create_stump(S, attribute_dict, labels=None, labeled=False, dtype=default_dtype, weights=None):
  '''Return a weak decision stump.
  '''
  return dt.ID3(S, attribute_dict, labels=labels, labeled=labeled, dtype=dtype, max_depth=1, weights=weights)


def AdaBoost(T, S, attribute_dict, labels=None, labeled=False, dtype=default_dtype):
  '''Return a classifier via AdaBoost algorithm using decision stumps as weak classifiers.
  Information Gain (using entropy) is used as gain metric.
  Input labels must be in {-1, 1}.

  Arguments:
  T -- iteration threshold
  S -- example data, no header.

  Keyword arguments:
  labels -- list of labels (Default: None)
  labeled -- is data labeled? if so must set to True (Default: False)
  dtype -- numpy dtype
  '''
  labels = dt.check_labels(S, labels=labels, labeled=labeled)
  stumps = []
  alphas = []

  m = np.shape(S)[0]
  Dt = [1/m for i in range(m)]
  for t in range(T):
    labels.fractional_counts = Dt
    stump = create_stump(S, attribute_dict, labels=labels, labeled=labeled, dtype=dtype, weights=None)
    stumps.append(stump)
    et = weighted_error(stumps[t], S, list(attribute_dict.keys()), labels=labels, labeled=labeled, weights=None)
    alphas.append( np.log((1-et)/et) /2 )

    print(stumps[t], stumps[t].branches)
    print(labels.fractional_counts[0:6])
    print(t, stumps[t])
    print("as: ", alphas[t])
    print(et)
    print()

    Dtt = [None]*len(Dt)
    print("pasing: ", list(S[1,:]))
    for i in range(len(Dt)):
     Dtt[i] = Dt[i] * np.exp( -alphas[t] * float(labels[i]) * float(dt.classify(stumps[t], list(S[i,:]), list(attribute_dict.keys()))))

    for i in range(len(Dtt)):
      Dt[i] = Dtt[i]/np.sum(Dtt)


  return stump
