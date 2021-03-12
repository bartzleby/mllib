#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# March, 2021
# 
# HW2 bank data set
# predict from first tree
# of each of forests
#

import numpy as np
import pickle

import data.bank.bankData as bd
from DecisionTree import classify
from DecisionTree import numeric2median
from EnsembleLearning import *


def main():
  dtype = bd.dtype
  attribute_dict = bd.attribute_dict

  with open('./pickle/experiment_forests_m200.pkl', 'rb') as file:
    bags = pickle.load(file)

  train = np.genfromtxt("../data/bank/train.csv", delimiter=',', dtype=dtype)
  test = np.genfromtxt("../data/bank/test.csv", delimiter=',', dtype=dtype)

  train[:,-1] = labels_to_pmone(train[:,-1])
  test[:,-1] = labels_to_pmone(test[:,-1])

  S, attribute_dict = numeric2median([train, test], attribute_dict);
  train = S[0]
  test = S[1]

  st_biases = []
  st_svars  = []

  tb_biases = []
  tb_svars  = []

  for tsti in range(np.shape(test)[0]):
    print(tsti)
    st_predictions = []
    tb_predictions = []
    for bag in bags:
      st_predictions.append(classify_from_tree_bag([bag[0]], test[tsti,0:-1], list(attribute_dict.keys())))
      tb_predictions.append(classify_from_tree_bag(bag, test[tsti,0:-1], list(attribute_dict.keys())))

    st_predictions = labels_to_pmone(st_predictions)
    tb_predictions = labels_to_pmone(tb_predictions)

    st_avg_pred = np.average(st_predictions)
    st_biases.append( (st_avg_pred - int(test[tsti,-1]))**2 )
    st_svars.append( np.var(st_predictions) )

    tb_avg_pred = np.average(tb_predictions)
    tb_biases.append( (tb_avg_pred - int(test[tsti,-1]))**2 )
    tb_svars.append( np.var(tb_predictions) )


  avg_bias_st = np.average(st_biases)
  avg_svar_st = np.average(st_svars)
  expd_sse_st = avg_bias_st + avg_svar_st

  avg_bias_tb = np.average(tb_biases)
  avg_svar_tb = np.average(tb_svars)
  expd_sse_tb = avg_bias_tb + avg_svar_tb


  print("single trees:")
  print("bias: ", avg_bias_st)
  print("svar: ", avg_svar_st)
  print("esse: ", expd_sse_st)
  print()

  print("bagged learners:")
  print("bias: ", avg_bias_tb)
  print("svar: ", avg_svar_tb)
  print("esse: ", expd_sse_tb)
  print()

  with open('./pickle/experiment_prediction_forest_stats_m200.pkl', 'wb') as file: 
    pickle.dump((avg_bias_st, avg_svar_st, expd_sse_st), file)
    pickle.dump((avg_bias_tb, avg_svar_tb, expd_sse_tb), file)

  return


if __name__ == '__main__':
    main()


