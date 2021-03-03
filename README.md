# mllib
This is a machine learning library developed by Danny Bartz 
for CS5350/6350 in University of Utah.


To learn a decision tree you will have to import the ID3 algroithm
from the DecisionTree module: `from DecisionTRee import ID3`.
Type `help(ID3)` from an interactive python3 command line to view usage.

note: in order to import from the DecisionTree module, the path will
need to be included in the PYTHONPATH environment variable.  This was
acheived in my case by adding the following to `/.cshrc` on CADE:

`setenv PYTHONPATH "/home/u6021420/cs-5350-ml/mllib"`

TODO: create venv to track dependencies.

TODO: incorporate pandas DataFrames?