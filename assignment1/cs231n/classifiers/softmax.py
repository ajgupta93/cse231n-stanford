import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  delta = 0.0000000001
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
    x = X[i,:].reshape(-1,1)
    prods = np.zeros((W.shape[1]))
    max_prod = -np.inf
    for c in range(W.shape[1]):
        w_class = W[:,c].reshape(-1,1)
        prods[c] = w_class.T.dot(x)
        if prods[c]>max_prod:
            max_prod = prods[c]
    prods-=max_prod
    num = np.power(np.e,prods[y[i]])
    denom = np.sum(np.power(np.e,prods))
    loss+=-np.log(num/denom)
    p = np.power(np.e,prods)/denom
    p[y[i]]-=1
    for c in range(W.shape[1]):
        dW[:,c] += p[c]*x.ravel()
  loss += 0.5*reg*np.sum(W**2)
  loss/=X.shape[0]
  dW/=X.shape[0]
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  wx = W.T.dot(X.T)
  wxmax = wx.max(0).reshape(1,-1)
  wx -= wxmax
  nx = np.power(np.e,wx)
  dx = nx.sum(0).reshape(1,-1)
  px = np.divide(nx,dx)
  ym = np.zeros((W.shape[1],X.shape[0]))
  ym[y,range(X.shape[0])] = 1
  loss = -np.sum(np.log(px[y,range(X.shape[0])]))/X.shape[0]+0.5*reg*np.sum(W**2)
  dW = ((px-ym).dot(X)).T/X.shape[0]+reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
