#
#
#
#
#
# note: as of python3.7, dictionaries are insertion ordered.

import DecisionTree as tree
from DecisionTree import DecisionTreeNode as dtn

import numpy as np
default_dtype = np.int8


class Labels(list):
  """Wrapper for python list class.
  """

  def dict(self):
    """Return a dictionary with possible label values as keys
    and count of each label value as dict value.
    """
    label_dict = {}
    for k in self:
      if k in label_dict:
        label_dict[k] += 1
      else:
        label_dict[k] = 1

    return label_dict

  def entropy(self):
    """Return entropy of labels.
    """
    label_dict = self.dict()
    H = 0
    for label, count in label_dict.items():
      H -= (count/len(self))*np.log2(count/len(self))

    return H

  def most_common(self):
    """Return most common label.
    """
    return max(set(self), key=self.count)


def Entropy(L):
  """Function to return the entropy of
  a set L of examples, including
  labels as rightmost column.

  Can also accept a one dimensional
  array of labels as input L, as input

  returns -1 if dimension of L > 2
  """
  if type(L) is list:
    return Labels(L).entropy()
  if type(L) is Labels:
    return Labels(L).entropy()

  if L.ndim > 2:
    return -1
  if L.ndim == 2:
    L = L[:,-1]

  return Entropy(list(L))


def get_attr_values_dict(S, a):
  """Returns a dictionary with each value attribute
  at col index a can take as keys, and lists of
  row indices where the attribute takes the key 
  value as dict values.
  """
  values_dict = {}
  for i in range(S.shape[0]):
    if S[i, a] in values_dict:
      values_dict[S[i, a]].append(i)
    else:
      values_dict[S[i, a]] = [i]

  return values_dict

def get_attr_dict(S, attributes=None, header=False):
  """Returns a dictionary with attribute names as keys
  and lists of possible attribute values as dict values.

  Keyword arguments:
  attributes -- list of attribute names (default None)
  header -- does data S have header listing attribute names? (default False)
  """
  if attributes is None and header:
    attributes = list(S[0,:])
    S = S[1:,:]
  elif attributes is None:
    attributes = [a for a in range(S.shape[1])]
  
  attr_dict = {}
  for a, attr in enumerate(attributes):
    attr_values = list(get_attr_values_dict(S, a).keys())
    attr_dict.update({attr: attr_values})

  return attr_dict


def get_Sv(S, a, v, labels, dtype=default_dtype):
  """Return new data and labels corresponding
  to rows of S where attribute at col index a
  takes value v.
  """

  # we collect all values A takes in S, 
  # and track row indices:
  values_dict = get_attr_values_dict(S, a)
  indices = values_dict[v]

  Sv = np.empty((len(indices), S.shape[1]), dtype=dtype)
  Sv_labels = Labels([])

  for i, r in enumerate(indices):
    Sv[i,:] = S[r,:]  
    Sv_labels.append(labels[r])

  return Sv, Sv_labels

def Gain(S, a, labels=None, labeled=False):
  """Function to return the information gain of 
  partitioning set S on attribute A, an index
  such that S[i, A] is Value(A) for example i.
  
  rows in S are examples, with S[i, -1]
  being the label for example i

  Returns negative one if no labels

  Arguments:
  S -- example data, no header
  a -- col index of attribute

  Keyword arguments:
  labels -- list of labels (Default: None)
  labeled -- is data labeled? if so must set to True (Default: False)
  """
  if labels is None and not labeled:
    return -1
  elif labeled:
    labels = list(S[:,-1])
    S = S[:,0:-1]

  labels = Labels(labels)

  nex = S.shape[0] # number of examples
  HS = labels.entropy()

  # we collect all values A takes in S, 
  # and track row indices:
  values_dict = get_attr_values_dict(S,a)

  G = HS
  for value, indices in values_dict.items():
    Sv_labels =  Labels([])
    for i, r in enumerate(indices):
      Sv_labels.append(labels[r])

    G -= (len(indices)/nex)*Sv_labels.entropy()

  return G


def ID3(S, attribute_dict, labels=None, labeled=False, dtype=default_dtype):
  """Contruct a decision tree via ID3 algorithm.

  Arguments:
  S -- example data, no header

  Keyword arguments:
  attributes -- list of attribute names (defalut None: attrs are numbered)
  labels -- list of labels (Default: None)
  labeled -- is data labeled? if so must set to True (Default: False)
  """

  if labels is None and not labeled:
    return
  elif labeled:
    labels = list(S[:,-1])
    S = S[:,0:-1]

  labels = Labels(labels);

  if Entropy(labels) == 0:
    nde = dtn()
    nde.label = labels[0]
    return nde

  attributes = list(attribute_dict.keys())
  if len(attributes) == 0:
    nde = dtn()
    nde.label = labels.most_common()
    return nde


  gains = [Gain(S, i, labels) for i in range(len(attributes))]
  #for i, g in enumerate(gains):
  #  print(attributes[i], g)

  max_gain = max(gains)
  mgi = gains.index(max_gain)
  split_attr = attributes[mgi]

  root = dtn(split_attr, attribute_dict[split_attr])

  for attr_val in root.values:
    Sv, Sv_labels = get_Sv(S, mgi, attr_val, labels, dtype=dtype)
    
    if np.size(Sv) == 0:
      root.branches.update({attr_val: dtn(label=labels.most_common())})

    else:
      Sv = np.delete(Sv, mgi, 1)
      new_attr_dict = duplicate_less(split_attr, attribute_dict)

      nde = ID3(Sv, new_attr_dict, Sv_labels, dtype=dtype)
      root.branches.update({attr_val: nde})


  return root



def duplicate_less(key, dict):
  """Duplicate a dictionary,
  leaving out key,
  without setting it equal 
  to the same object.
  """
  new_dict = {}

  for k, v in dict.items():
   if k != key:
    new_dict.update({k: v})

  return new_dict
