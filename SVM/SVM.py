#!/usr/bin/env python3

# 
# CS5350
# Danny Bartz
# April, 2021
# 
# HW4 Support Vector Machines
#

import numpy as np


def SVM_objective(C, S, w):
  '''Calculate the value of the SVM objective function.
  Assumes bias parameter is prepended on weight vector.
  Assumes S is labeled on {-1, 1}.
  '''
  sum = 0
  for i in range(np.shape(S)[0]):
    sum += max(0, 1 - S[i,-1]*(w@S[i,0:-1]))

  return (w[1:]@w[1:])/2 + C*sum


def SVM_P_SSGD(T, S, C, gamma_0=1, schedule=None, retobj=False):
  '''Run stochastic sub-gradient descent algorithm
  for Support Vector Machine in primal domain
  for T epochs.
  Returns tuple containing learned weight vector and
    SVM objective function value at each t in a list.
   (if retobj, otherwise just weight vector)

  Arguments:
    T -- number of epochs to run
    S -- example data, prepended with ones,
            labeled in {-1, 1}
    C -- 

  Keyword Arguments:
    gamma_0 -- initial learning rate (Default: 1)
    schedule -- Learning Rate Schedule Function (Default: None)
                signature: schedule(gamma_0, t, d)
    retobj -- return SVM objective values at each step too?
                                            (Default: False)
  '''
  m = np.shape(S)[0]

  w = np.zeros(np.shape(S)[1]-1)
  w_0 = np.ones(np.shape(w))
  w_0[0]=0

  d = 1
  t = 1
  gamma_t = gamma_0;
  if retobj:
    ovals = [SVM_objective(C, S, w)]
  for i in range(T):
    np.random.shuffle(S);
    for i in range(m):
      y_i = S[i,-1]
      x_i = S[i,0:-1]
      if schedule is not None:
        gamma_t = schedule(gamma_0, t, d)
      t += 1

      if y_i*(w@x_i) <= 1:
        w = w - gamma_t*w*w_0 + gamma_t*C*m*y_i*x_i
      else:
        w[1:] = (1-gamma_t)*w[1:]

      if retobj:
        ovals.append(SVM_objective(C, S, w))

  if retobj:
    return (w, ovals)
  return w



def main():
  return

if __name__ == '__main__':
    main()


