#! /bin/sh

mkdir DecisionTree/pickle
mkdir EnsembleLearning/pickle
mkdir LinearRegression/pickle


cd LinearRegression
python3 test.py
cd ..


cd EnsembleLearning

python3 adaboostBank.py
python3 adaboostBankErrors.yp
python3 adaboostBankErrorsStumpwise.py 
#daboostBankPlot.py

python3 baggedBank.py
python3 baggedBankErrors.py
#aggedBankPlot.py

python3 randomforestBank.py
python3 randomforestBankErrors.py
#andomforestBankPlot.py

python3 experiment_genbags.py
python3 experiment_genforest.py
python3 experiment_predict.py
python3 experiment_predict_forest.py
python3 experiment_show.py
python3 experiment_show_forest.py pickle/experiment_prediction_forest_stats_m500.pkl

python3 def_credit_run_100.py
python3 def_credit_run_trees.py
python3 def_credit_errors.py

cd ../LinearRegression

python3 conc_GDB.py
python3 cond_GDS.py
python3 conc_res.py

cd ..
