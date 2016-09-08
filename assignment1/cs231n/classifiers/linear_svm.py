import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # dW has the same shape as W, D x C
        dW[:, j] += X[i].T
        dW[:, y[i]] -= X[i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  
  # divide weight by num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Also regularize dW
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = np.dot(X, W) # size N x C

  # !!!! It is not that beautiful, I will find a way to fix it !!!!
  # try to extend the size of correct scores from Nx1 to NxC in order to do the following excute
  # use CxN ones multiply 1xN to do broadcasting and then Transpose it as NxC
  correct_scores = np.ones([scores.shape[1], scores.shape[0]]) * scores[np.arange(0, scores.shape[0]), y]
  correct_scores = correct_scores.T

  delta = np.ones(scores.shape)
  margin = scores - correct_scores + delta
  # remove margin element which is less than 0
  margin[margin < 0] = 0
  # set y_i in margin as 0, cause we don't count it accutually
  margin[np.arange(0, scores.shape[0]), y] = 0
  loss = np.sum(margin)
  # Average loss
  loss /= num_train
  # add regularization term
  loss += 0.5 * reg * np.sum(W * W)  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margin = scores - correct_scores + delta # size NxC
  margin[margin < 0] = 0
  # only sums up all those meet the condition
  # and remember not to count y_i
  margin[margin > 0] = 1
  margin[np.arange(0, scores.shape[0]), y] = 0

  # set y_i as the total number of the row, and it should be negative
  margin[np.arange(0, scores.shape[0]), y] = -np.sum(margin, axis=1)
  dW = np.dot(X.T, margin)
  # Average dW
  dW /= num_train
  # Add regularization term
  dW += reg * W    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
