#
#
#
#
#
#
#
#



import numpy as np

from DecisionTree import *


default_dtype = np.int8


def create_stump(S, attribute_dict, labels=None, labeled=False, dtype=default_dtype, gain_metric='entropy'):
  '''
  '''
  stump =  ID3(S, attribute_dict, labels=labels, labeled=labeled, dtype=dtype, gain_metric=gain_metric, current_depth=0, max_depth=1)

  return stump


def main():
  print('here.')

  return 0


if __name__ == '__main__':
    main()
