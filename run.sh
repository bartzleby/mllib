#! /bin/sh


if [ ! -d SVM/pickle ] 
then
  mkdir SVM/pickle
fi


cd SVM
python3 run_p5.py
python3 run_SVM_SSGD.py
python3 test.py

python3 run_SVM_dual_optimize.py
python3 run_SVM_dual_test.py



cd ..
