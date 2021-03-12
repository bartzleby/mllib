#
#
#
#
#
# Note: as of python3.7, dictionaries are insertion ordered.

import DecisionTree as tree
from DecisionTree import DecisionTreeNode as dtn

import random
import numpy as np
default_dtype = np.int8

from collections import Counter
from statistics import mode

class MissingLabelsError(Exception):
    """Exception raised for missing labels
    within a function.
    (not passed properly)

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Potentially missing labels!"):
        self.message = message
        super().__init__(self.message)


class Labels(list):
  """Wrapper for python list class.
  """

  def __init__(self, base_list, fractional_counts=None):
    super().__init__(base_list)
    if type(base_list) is not Labels:
     if fractional_counts is None:
      self.fractional_counts = np.ones(len(self))
     else:
      self.fractional_counts = fractional_counts; # TODO: check length

    else:
      self.fractional_counts = base_list.fractional_counts

    self.size = sum(self.fractional_counts)

  def dict(self):
    """Return a dictionary with possible label values as keys
    and count of each label value as dict value, taking into 
    account fractional counts.
    """
    label_dict = {}
    for i, k in enumerate(self):
      if k in label_dict:
        label_dict[k] += self.fractional_counts[i]
      else:
        label_dict[k] = self.fractional_counts[i]

    return label_dict

  def entropy(self, base=np.e):
    """Return entropy of labels.
    """
    label_dict = self.dict()
    H = 0
    for label, count in label_dict.items():
      H -= (count/self.size)*np.log(count/self.size)/np.log(base)

    return H

  def majority_error(self):
    """Return majority error of labels, ie the error if
    the most common label were chosen to represent all.
    """
    label_dict = self.dict()
    return (self.size - label_dict[self.most_common()])/self.size

  def Gini_index(self):
    """
    """
    label_dict = self.dict()
    GI = 1
    for c in label_dict.values():
      GI -= (c/self.size)**2

    return GI


  def most_common(self):
    """Return most common label.
    Accounting fractional examples.
    """
    label_dict = self.dict()
    counts = list(label_dict.values())
    mci = counts.index(max(counts))
    attrs = list(label_dict.keys())
    return attrs[mci]

  def assign_fractional_counts(self, S, missing_values='?'):
    '''Assign fractional counts to labels
    based on data set S with missing values.

    It is assumed that the data set S 
    corresponds to these labels.

    updates self and returns new data set.

    Arguments:
      S -- Dataset, unlabeled

    Keyword arguments:
      missing_values -- string representing missing values
                        to be filled with fractional counts
    '''
    example_count = len(S[:,0])

    attr_props = []
    # first we gather the attribute proportions
    # from original data set
    for a in range(len(S[0,:])):
      apd = {}
      ad = get_attr_values_dict(S, a)
      for attr_val, cols in ad.items():
       if missing_values in ad.keys() and attr_val != missing_values:
        apd.update({attr_val: len(cols)/(example_count-len(ad[missing_values]))})
       else:
        apd.update({attr_val: len(cols)/example_count})
      attr_props.append(apd)

    # then we traverse the data set and to find rows
    # with missing values, and cols for which it is so
    rowDict = get_missing_value_locations(S, missing_values)


    # and pop these rows from S and from self.
    # we go in reverse so indices from rowDict
    # match after each pop operation.
    popped_labels = []
    popped_labels_fcs = []
    popped_data_rows = np.empty((len(rowDict), len(S[0,:])), dtype=S.dtype)
    for i, poprow in enumerate(reversed(list(rowDict.keys()))):
      popped_labels.insert(0, self.pop(poprow))
      popped_labels_fcs.insert(0, self.fractional_counts[poprow])
      self.fractional_counts = np.delete(self.fractional_counts, poprow)
      popped_data_rows[len(rowDict)-i-1, :] = S[poprow,:]
      S = np.delete(S, poprow, axis=0)


    ####################################################
    concat_data = []
    for i in range(len(popped_labels)):
      new_list = list(popped_data_rows[i,:])
      new_list.append(popped_labels[i])
      new_list.append(popped_labels_fcs[i])
      concat_data.append(new_list)


    # TODO:
    # clean this up, and maybe use recursion?
    new_examples_collection = []    
    # things can get a bit messy if dealing with
    # multiple missing attributes on an example
    # we basically need to do a sort of cross
    # product, ...
    for r, cols in enumerate(list(rowDict.values())):
      new_examples = []
      for attri in cols:
        if new_examples == []:
          #create a new row for each val attr can take:
          for v in list(attr_props[attri].keys()):
            if v != missing_values:
              new_row = list(concat_data[r])
              new_row[attri] = v
              new_row[-1] = concat_data[r][-1]*attr_props[attri][v]
              new_examples.append(new_row)
          if len(cols) == 1:
            for ex in new_examples:
               new_examples_collection.append(ex)

        else:
          nnew_examples = []
          for example in new_examples:
            #create a new row for each val attr can take:
            for v in list(attr_props[attri].keys()):
              if v != missing_values:
                new_row = list(example)
                new_row[attri] = v
                new_row[-1] = example[-1]*attr_props[attri][v]
                nnew_examples.append(new_row)
                
          new_examples.clear()
          new_examples = list(nnew_examples)

      if len(cols) > 1:
        for ex in new_examples:
          new_examples_collection.append(ex)


    mergarr = np.empty((len(new_examples_collection), len(S[0,:])), dtype=S.dtype)
    fcl = list(self.fractional_counts)
    for i, ne in enumerate(new_examples_collection):
      self.append(ne[-2])
      fcl.append(ne[-1])
      mergarr[i,:] =  ne[0:4]

    self.fractional_counts = np.array(fcl)
    self.size = sum(self.fractional_counts)

    return np.append(S, mergarr, 0)

def get_missing_value_locations(S, missing_values='?'):
  '''Returna s dict with rows containing missing
  values as keys, and lists of cols where they
  occur in the row.
  '''
  rowDict = {}
  for r  in range(len(S[:,0])):
    rl = list(S[r,:])
    if missing_values in rl:
      wi = [i for i,x in enumerate(rl) if x==missing_values]
      rowDict.update({r: wi})

  return rowDict

def assign_most_common_general(S, missing_values='?', mcvs=None):
  '''Assign to missing attribute values
  the most common value from data set
  at large.

  returns new data set.

  Arguments:
    S -- Data Set
  Keyword arguments:
    missing_values -- string representing missing values
                      to be filled with fractional count
    mcvs -- most common values if known
  '''
  if mcvs is None:
    mcvs = [mode(S[:,a]) if mode(S[:,a]) != missing_values                 \
                         else Counter(S[:,a].ravel()).most_common(2)[1][0] \
                         for a in range(len(S[0,:]))                       ]

  mvloc = get_missing_value_locations(S, missing_values=missing_values)
  for r, cs in mvloc.items():
   for c in cs:
    S[r,c] = mcvs[c]

  return S, mcvs


def assign_most_common_specific(S):
  '''Assign to missing attribute values
  the most common value from data set
  where the label matches.
  
  Not yet implemented.
  '''
  return NotImplementedError


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


def get_Sv_labels(labels, indices):
  '''
  '''
  Sv_labels = Labels([])
  lfcs = []

  for i, r in enumerate(indices):
    Sv_labels.append(labels[r])
    lfcs.append(labels.fractional_counts[r])

  Sv_labels.fractional_counts = np.array(lfcs)
  Sv_labels.size = sum(Sv_labels.fractional_counts)

  return Sv_labels


def get_Sv(S, a, v, labels, dtype=default_dtype):
  """Return new data and labels corresponding
  to rows of S where attribute at col index a
  takes value v.
  """

  # we collect all values A takes in S, 
  # and track row indices:
  values_dict = get_attr_values_dict(S, a)
  try:
    indices = values_dict[v]
  except KeyError:
    return np.empty(shape=(0,0)), Labels([])

  Sv_labels = get_Sv_labels(labels, indices)
  Sv = np.empty((len(indices), S.shape[1]), dtype=dtype)
  for i, r in enumerate(indices):
    Sv[i,:] = S[r,:]

  return Sv, Sv_labels

def check_labels(S, labels=None, labeled=False):
  '''Does passing np arr hinder performance?
  '''
  if labels is None and not labeled:
    raise MissingLabelsError()
  elif labeled:
    labels = list(S[:,-1])
    S = S[:,0:-1]

  return Labels(labels)


def Gain(S, a, labels=None, labeled=False, metric="entropy", base=np.e):
  """Function to return the information gain of 
  partitioning set S on attribute a, an index
  such that S[i, a] is Value(A) for example i.
  
  rows in S are examples, with S[i, -1]
  being the label for example i

  Returns negative one if no labels

  Arguments:
  S -- example data, no header
  a -- col index of attribute

  Keyword arguments:
  labels -- list of labels (Default: None)
  labeled -- is data labeled? if so must set to True (Default: False)
  gain_metric -- what is used to calculate gain? (Default: entropy)
      - entropy
      - majority_error
      - Gini_index
  """
  labels = check_labels(S, labels=labels, labeled=labeled)
  nex = labels.size # number of examples

  G = 0
  # we collect all values a takes in S, 
  # and track row indices:
  values_dict = get_attr_values_dict(S,a)

  if metric == "entropy":
    G = labels.entropy(base=base)
    for value, indices in values_dict.items():
      Sv_labels = get_Sv_labels(labels, indices)
      G -= (Sv_labels.size/nex)*Sv_labels.entropy(base=base)

  elif metric == "majority_error":
    G = labels.majority_error()
    for value, indices in values_dict.items():
      Sv_labels = get_Sv_labels(labels, indices)
      G-= (Sv_labels.size/nex)*Sv_labels.majority_error()

  elif metric == "Gini_index":
    G = labels.Gini_index()
    for value, indices in values_dict.items():
      Sv_labels = get_Sv_labels(labels, indices)
      G-= (Sv_labels.size/nex)*Sv_labels.Gini_index()

  return G


def ID3(S, attribute_dict, labels=None, labeled=False, dtype=default_dtype, gain_metric="entropy", current_depth = 0, max_depth=np.inf, RandTree=False, NumRandAttr=1, display=False):
  """Contruct a decision tree via ID3 algorithm.

  Arguments:
  S -- example data, no header

  Keyword arguments:
  attributes -- list of attribute names (defalut None: attrs are numbered)
  labels -- list of labels (Default: None)
  labeled -- is data labeled? if so must set to True (Default: False)
  dtype -- 
  gain_metric -- what is used to calculate gain? (Default: entropy)
      - entropy
      - majority_error
      - Gini_index

  current_depth -- 
  max_depth -- maximum depth of resulting tree (Default: no limit)
  NumRandAttr -- number of random attributes to consider if RandTree (Default: 1)
  RandTree -- select a small random subset of features to split at each node?
                                                  (Default: False)
  """
  labels = check_labels(S, labels=labels, labeled=labeled)

  if current_depth == max_depth:
    nde = dtn()
    nde.label = labels.most_common()
    return nde

  if labels.entropy() == 0:
    nde = dtn()
    nde.label = labels[0]
    return nde

  attributes = list(attribute_dict.keys())
  if len(attributes) == 0:
    nde = dtn()
    nde.label = labels.most_common()
    return nde

  # intercept here if needed (num2med)

  if RandTree:
    featureIndices = random.sample(range(len(attributes)), NumRandAttr)
    featureIndices.sort()
  else:
    featureIndices = range(len(attributes))

  gains = [Gain(S, i, labels, metric=gain_metric) for i in featureIndices]

  max_gain = max(gains)
  mgi = gains.index(max_gain)
  split_attr = attributes[mgi]

  if display:
   D = np.empty((np.shape(S)[0], np.shape(S)[1]+2), dtype=S.dtype)
   D[:,0:-2] = S
   D[:,-2] = labels
   D[:,-1] = labels.fractional_counts
   print("labeled data with fractional counts: ")
   print(D)
   print("entropy of data: ", labels.entropy())
   print("gains per attribute (may not be information gain): ")
   for i, a in enumerate(attributes):
    print(a, Gain(S, i, labels=labels, metric=gain_metric))
   print("We split on: ", split_attr, '\n')


  root = dtn(split_attr, attribute_dict[split_attr])

  for attr_val in root.values:
    Sv, Sv_labels = get_Sv(S, mgi, attr_val, labels, dtype=dtype)

    if np.size(Sv) == 0:
      root.branches.update({attr_val: dtn(label=labels.most_common())})

    else:
      Sv = np.delete(Sv, mgi, 1)
      new_attr_dict = duplicate_less(split_attr, attribute_dict)

      nde = ID3(Sv, new_attr_dict, Sv_labels, dtype=dtype, current_depth=1+current_depth, max_depth=max_depth, display=display)
      root.branches.update({attr_val: nde})


  return root


def numeric2median(SS, attribute_dict):
  '''here we intercept 'numeric' attributes and convert them to binary on median.
  returns list of ndarrays corresponding to input SS.

  Arguments:
    SS -- a set or list of two data sets, training data first
         (test data is filled based on median from training set)
    attribute_dict -- 
  '''
  RSS = [] # return set
  # TODO: preserve the original numbers and convert them again on subset median?
  numeric_indices = [i for i,x in enumerate(list(attribute_dict.values())) if x=='numeric']
  attr_medians = []
  for ni in numeric_indices:
    attr_vals = SS[0][:,ni]
    attr_medians.append(np.median(attr_vals.astype(np.float), axis=0))

  for S in SS:
   for i, ni in enumerate(numeric_indices):
    for r in range(np.shape(S)[0]):
     S[r,ni] = 'yes' if S[r,ni].astype(np.float) > attr_medians[i] else 'no'
   
   RSS.append(S)

  # then we must update the attribute_dict to reflect the change
  for i in numeric_indices:
    attribute_dict.update({list(attribute_dict.keys())[i]:['no','yes']})

  return RSS, attribute_dict

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
