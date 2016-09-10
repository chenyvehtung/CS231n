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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #print X.shape, W.shape
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in xrange(num_train):
    f = np.dot(X[i], W)
    # prevent from numeric instability
    f_max = np.max(f)
    f -= f_max
    f = np.exp(f)
    f_sum = np.sum(f)
    loss -= np.log(f[y[i]] / f_sum)
    # dW in size D x C
    for j in xrange(num_class):
      dW[:, j] += 1 / f_sum * f[j] * X[i]
      if j == y[i]:
        dW[:, j] -= X[i]
  # Average the loss and dW
  loss /= num_train
  dW /= num_train
  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W      
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
  # W in size (D, C), while X in size (N, D)
  f = X.dot(W)
  f_max = np.max(f, axis=1)

  # for broadcasting
  f = f.T 
  f_max = f_max.T
  f -= f_max
  f = f.T

  f = np.exp(f)
  f_sum = np.sum(f, axis=1)
  f_correct_class = f[np.arange(f.shape[0]), y]
  loss -= np.sum(np.log(f_correct_class / f_sum)) 

  # for broadcasting
  f = f.T
  f_sum = f_sum.T
  f /= f_sum
  f = f.T

  # for correct class
  f[np.arange(f.shape[0]), y] -= 1
  dW = np.dot(X.T, f)

  # Average 
  loss /= X.shape[0]
  dW /= X.shape[0]
  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

