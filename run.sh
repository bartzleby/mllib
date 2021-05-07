#! /bin/sh

if [ ! -d LogisticRegression/pickle ] 
then
  mkdir LogisticRegression/pickle
fi

cd LogisticRegression
python3 run_MAP_p4.py
python3 run_MAP_banknote.py
python3 run_MLE_banknote.py
python3 plot_mle.py
python3 plot_map.py
python3 test_mle.py
python3 test_map.py

cd ../NeuralNetworks
python3 run_network_banknote.py
python3 run_banknote_torch_net_ReLU.py

cd ..
