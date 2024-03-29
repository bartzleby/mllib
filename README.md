# mllib
This is a machine learning library developed by Danny Bartz 
for CS5350/6350 in University of Utah.

note: in order to import from modules in this library, the path will
need to be included in the PYTHONPATH environment variable.  This was
acheived in my case by adding the following to `~/.cshrc` on CADE: <br />
`setenv PYTHONPATH "/home/u6021420/cs-5350-ml/mllib"`

Functions are generally well documented, and usage instructions
can be viewed by typing `help({function_name})` from an interactive
python3 command line.


HW4: SVM
To use these function, you must import 



HW3: Perceptron

Perceptron algorithms are divided into two functions <br />
within the Perceptron package, which can be imported <br />
with `import Perceptron`.  These function are: <br />
`Perceptron` and `VotedPerceptron` <br />
`Perceptron` can be run in `standard` mode, or in `average` mode <br />
`VotedPerceptron` implements the voted perceptron algorithm. <br />
A prediction can be obtained from learned weight vectors  <br />
and prediction counts using `PredictVoted`. <br />
Type `help({function})` from a python3 interactive session <br />
to view more detailed instructions.



HW2:

The regression on concrete SLUMP data is performed in the 
`mllib/LinearRegression` directory.  The results are saved
as a pickle file in the pickle directory with a filename 
that matches the script which produced it.  To perform batch 
descent, simply run `python3 conc_GDB.py` in this directory.
The `conc_GDS.py` script performs stochastic gradient descent.
Data visualization is handled in `conc_res.py`.


To learn a decision tree you will have to import the ID3 algroithm
from the DecisionTree module: `from DecisionTree import ID3`.
Type `help(ID3)` from an interactive python3 command line to view usage.

The AdaBoost algorithm is implemented in `EnsembleLearning.py`
within the `mllib/EnsembleLearning` directory.

For the implementation of weights in the AdaBoost algorithm, we
simply repurpose the fractional_counts field of the Label class.

The data processing and visualization is split into three steps: <br />
 1 -- learning the AdaBoost hypothesis `adaboostBank.py` <br />
 2 -- calculating test and training errors 'adaboostBankErrors.py' <br />
 3 -- plotting these errors `adaboostBankPlot.py` <br />

After each step, relevant variables are saved in .pkl files
to be loaded at the next step.

A similar workflow is adopted for bagging trees and random forest generation.
To view usage instructions, type `help(AdaBoost)`, `help(BaggedTrees)`, or
`help(RandomForest)`.

TODO: create weights parallel to fractional_counts <br />
TODO: create venv to track dependencies. <br />
TODO: incorporate pandas DataFrames? <br />

-------------------------------------------------------------------------