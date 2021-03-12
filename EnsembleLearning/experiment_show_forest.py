#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 stat experiment
# show results
# 
# usage:
# python3 experiment_show_forest_stats.py path_to_pickle_file
#

import sys
import pickle

def main():

  with open(sys.argv[1], 'rb') as file:
    single_tree_stats = pickle.load(file) 
    bagged_tree_stats = pickle.load(file) 

  print(' & '.join([ '%f' % el for el in single_tree_stats]), end=' \\\\\n')
  print(' & '.join([ '%f' % el for el in bagged_tree_stats]), end=' \\\\\n')

  return


if __name__ == '__main__':
    main()


