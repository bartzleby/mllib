# mllib
This is a machine learning library developed by Danny Bartz 
for CS5350/6350 in University of Utah.


To learn a decision tree you will have to import the ID3 algroithm
from the DecisionTree module: `from DecisionTree import ID3`.
Type `help(ID3)` from an interactive python3 command line to view usage.

note: in order to import from the DecisionTree module, the path will
need to be included in the PYTHONPATH environment variable.  This was
acheived in my case by adding the following to `/.cshrc` on CADE: <br />
`setenv PYTHONPATH "/home/u6021420/cs-5350-ml/mllib"`

The AdaBoost algorithm is implemented in `EnsembleLearning.py`
within the `EnsembleLearning` directory.

For the implementation of weights in the AdaBoost algorithm, we
simply repurpose the fractional_counts field of the Label class.

The data processing and visualization is split into three steps:
 1 -- learning the AdaBoost hypothesis `adaboostBank.py` <br />
 2 -- calculating test and training errors 'adaboostBankErrors.py' <br />
 3 -- plotting these errors `adaboostBankPlot.py` <br />

After each step, relevant variables are saved in .pkl files
to be loaded at the next step.


TODO: create weights parallel to fractional_counts <br />
TODO: create venv to track dependencies. <br />
TODO: incorporate pandas DataFrames? <br />