#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 stat experiment
# show results
# 

import pickle

def main():

  with open('./pickle/experiment_prediction_stats.pkl', 'rb') as file:
    single_tree_stats = pickle.load(file) 
    bagged_tree_stats = pickle.load(file) 

  with open('./pickle/experiment_prediction_stats_m500.pkl', 'rb') as file:
    single_tree_stats_m500 = pickle.load(file) 
    bagged_tree_stats_m500 = pickle.load(file) 

  print(single_tree_stats)
  print(bagged_tree_stats)
  print(single_tree_stats_m500)
  print(bagged_tree_stats_m500)

  return


if __name__ == '__main__':
    main()


