#
#
#
#


import numpy as np


def lms_cost_w(y, w, xs):
  '''LMS cost function.
  '''
  J = 0
  for i in range(np.shape(xs)[0]):
    J += (y[i] - w@xs[i,:])**2

  return J/2


def grad_LMS(y, w, x, dtype=np.float):
  '''Calculate and return LMS gradient
  considering single example x.
  '''
  gradient = np.zeros(np.shape(w), dtype=dtype)
  for j in range(len(w)):
    gradient[j] -= ( y - w@x ) * x[j]

  return gradient

def grad_LMS_batch(y, w, xs, dtype=np.float):
  '''Calculate and return LMS gradient
  considering all data in xs.
  '''
  gradient = np.zeros(np.shape(w), dtype=dtype)
  for i in range(np.shape(xs)[0]):
    for j in range(len(w)):
      gradient[j] -= ( y[i] - w@xs[i,:] ) * xs[i][j]

  return gradient


def BatchGradientDescent():
  '''
  '''


  return NotImplementedError

def StochasticGradientDescent():
  '''
  '''


  return NotImplementedError



def LMS_Analytic(X, y):
  '''Return optimal weight vector given X, y.

  Arguments:
    X -- data matrix: [m rows of examples by d rows of features]
    y -- outcome vector
  '''
  return np.linalg.inv(X.transpose()@X)@X.transpose()@y
