
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





def classify_n_weak(n, H, example, attributes):
  '''Classify example using hypothesis H using first n
  weak classifiers from H.

  Returns int in {-1,1}

  Arguments:
    n -- number of weak classifiers to consider in classification
    H -- AdaBoost hypothesis, [alphas, dec.stumps]
    example -- example to be predicted on {-1,1}
    attributes -- list of attributes
  '''
  pred = 0
  for i in range(n):
    pred += float(H[0][i])*float(dt.classify(H[1][i], example, attributes))
  
  return np.sign(pred)




def classify_from_tree_bag(trees, example, attributes):
  '''Classify example using a bag (list) of decision trees.

  Returns predicted label

  Arguments:
    trees -- list of learned decision tree roots
    example -- example to be classified
    attributes -- list of attributes
  '''
  preds = {}
  for root in trees:
    pred = dt.classify(root, example, attributes)
    if pred in preds:
      preds[pred] += 1
    else:
      preds.update({pred: 1})
  
  majority_vote = max(preds, key=preds.get)
  return majority_vote


def draw_bootstrap_sample(m, S, labels=None, labeled=False, dtype=default_dtype):
  '''Draw m samples uniformly with replacement from data
  set S and associated labels.

  Returns [Sm, Lm]

  Arguments:
    n -- number of samples to return
    S -- dataset to be sampled

  Keyword Arguments:
  labels -- list of labels (Default: None)
  labeled -- is data labeled? if so must set to True (Default: False)
  '''
  labels = dt.check_labels(S, labels=labels, labeled=labeled)

  Sm = np.empty((m, np.shape(S)[1]), dtype=dtype)
  Lm = []; fcs = [];

  samples = np.random.randint(np.shape(S)[0],size=m)
  for i, s in enumerate(samples):
    Sm[i,:] = S[s,:]
    Lm.append(labels[s])
    fcs.append(labels.fractional_counts[s])

  return [Sm, dt.Labels(Lm, fractional_counts=fcs)]


def RandomForest(m, T, S, attribute_dict, labels=None, labeled=False, dtype=default_dtype, NumRandAttr=1):
  '''Return bag (list) of T trees learned by drawing
  a bootstrap sample of size m from data set S and 
  selecting random subset of features at each node.

  Arguments:
    m -- size of bootstrap samples
    T -- number of trees to produce
    S -- data set from which to sample
    attribute_dict -- 

  Keyword Arguments:
  labels -- list of labels (Default: None)
  labeled -- is data labeled? if so must set to True (Default: False)
  dtype -- numpy dtype
  NumRandAttr -- number of random attributes to consider at each node split (Default: 1)
  '''
  labels = dt.check_labels(S, labels=labels, labeled=labeled)
  if labeled:
    S = S[:,0:-1]

  forest = []
  for t in range(T):
    bss = draw_bootstrap_sample(m, S, labels=labels, dtype=dtype)
    forest.append(dt.ID3(bss[0], attribute_dict, labels=bss[1], dtype=dtype, RandTree=True, NumRandAttr=NumRandAttr))  

  return forest


def BaggedTrees(m, T, S, attribute_dict, labels=None, labeled=False, dtype=default_dtype):
  '''Return bag (list) of T trees learned by drawing
  a bootstrap sample of size m from data set S.

  Arguments:
    m -- size of bootstrap samples
    T -- number of trees to produce
    S -- data set from which to sample
    attribute_dict -- 

  Keyword Arguments:
  labels -- list of labels (Default: None)
  labeled -- is data labeled? if so must set to True (Default: False)
  dtype -- numpy dtype
  '''
  labels = dt.check_labels(S, labels=labels, labeled=labeled)
  if labeled:
    S = S[:,0:-1]

  trees = []
  for t in range(T):
    bss = draw_bootstrap_sample(m, S, labels=labels, dtype=dtype)
    trees.append(dt.ID3(bss[0], attribute_dict, labels=bss[1], dtype=dtype))


  return trees



def weighted_error(stump, S, attributes, labels=None, labeled=False):
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
  Dt = labels.fractional_counts

  et = 1/2
  for i in range(len(Dt)):
    try:
      et -= 0.5*Dt[i]*float(labels[i])*float(dt.classify(stump, list(S[i,:]), list(attributes)))
    except:
      print(labels[i])
      print(dt.classify(stump, list(S[i,:]), list(attributes)))
      return KeyError
  return et


def create_stump(S, attribute_dict, labels=None, labeled=False, dtype=default_dtype):
  '''Return a presumably weak decision stump.
  '''
  return dt.ID3(S, attribute_dict, labels=labels, labeled=labeled, dtype=dtype, max_depth=1)


def AdaBoost(T, S, attribute_dict, labels=None, labeled=False, dtype=default_dtype):
  '''Return a classifier via AdaBoost algorithm using decision stumps as weak classifiers.
  Information Gain (using entropy) is used as gain metric.
  Input labels must be in {-1, 1}.

  Returns list of lists: [Alphas[], Stumps[]]

  Arguments:
    T -- iteration threshold
    S -- example data, no header.
    attribute_dict -- 

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
    stumps.append(create_stump(S, attribute_dict, labels=labels, labeled=labeled, dtype=dtype))
    et = weighted_error(stumps[t], S, list(attribute_dict.keys()), labels=labels, labeled=labeled)
    alphas.append( np.log((1-et)/et) /2 )

    print(t, stumps[t], et)

    Dtt = [None]*len(Dt)
    for i in range(len(Dt)):
     Dtt[i] = Dt[i] * np.exp( -alphas[t] * float(labels[i]) * float(dt.classify(stumps[t], list(S[i,:]), list(attribute_dict.keys()))))

    for i in range(len(Dtt)):
      Dt[i] = Dtt[i]/np.sum(Dtt)


  return [alphas, stumps]


def labels_to_pmone(labels):
  '''map labels in {no, yes} to {-1,1} respectively.
  '''
  # TODO: probably a better way..
  for l in range(len(labels)):
    if labels[l] == 'no':
      labels[l] = -1
    elif labels[l] == 'yes':
      labels[l] = 1
    else:
      print('unexpected label!')
      print("label: ", labels[l])
      return

  return labels


def labels_to_pmone_01(labels):
  '''map labels in {0, 1} to {-1,1} respectively.
  '''
  # TODO: probably a better way..
  for l in range(len(labels)):
    if labels[l] == '0' or 0:
      labels[l] = -1
    elif labels[l] == '1' or 1:
      labels[l] = 1
    else:
      print('unexpected label!')
      print("label: ", labels[l])
      return

  return labels
