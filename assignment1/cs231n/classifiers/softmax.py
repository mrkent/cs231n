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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]  # or C
  num_train = X.shape[0]    # or N (10 labels)
  # compute the loss and the gradient

  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    lossexp = 0

    for j in xrange(num_classes):
      lossexp += np.exp(scores[j])

    loss += -correct_class_score + np.log(lossexp)

    for j in xrange(num_classes):
      if j == y[i]:
        dLdF = np.exp(scores[j])/lossexp - 1
      else:
        dLdF = np.exp(scores[j])/lossexp
      dW[:,j] += dLdF * X[i]



  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss +=  0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # DONE: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
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
  num_classes = W.shape[1]  # or C (10 labels)
  num_train = X.shape[0]    # or N
  scores = X.dot(W) # N x C matrix
  scores -= np.max(scores)

  # scores corresponding to target class in 1d array for each N
  correct_scores = scores[range(num_train), y]
  scores_sums = np.sum(np.exp(scores), axis=1)
  loss = -np.mean(np.log(np.exp(correct_scores)/scores_sums))

  dLdF = np.zeros((num_train, num_classes))
  dLdF = np.exp(scores)/scores_sums[:,np.newaxis]
  dLdF[range(num_train), y] -= 1

  dW = (X.T).dot(dLdF)

  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

