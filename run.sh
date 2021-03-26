#! /bin/sh


if [ ! -d Perceptron/pickle ] 
then
  mkdir Perceptron/pickle
fi


cd Perceptron
python3 run_perceptron.py
python3 test.py
cd ..
