from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_class = W.shape[1]
    
    for i in range(num_train):
        score = X[i].dot(W)
        score = score - np.max(score)
        score = np.exp(score)
        
        sumage = np.sum(score)
        corr = score[y[i]]
        loss -= np.log(corr/sumage)
        
        for j in range(num_class):
            if j == y[i]:
                continue
            dW[:,j] += score[j]/sumage * X[i]
        dW[:,y[i]] -= (sumage - corr) / sumage * X[i]
        
    loss /= num_train
    loss += reg*np.sum(W*W)
    
    dW /= num_train
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_class = W.shape[1]
    score = X.dot(W)
    score = score - np.max(score, axis = 1).reshape(num_train, 1)

    score = np.exp(score)

    corr = score[range(num_train), y].reshape(num_train, 1)
    sumage = np.sum(score, axis = 1).reshape(num_train, 1)
   
    loss -= np.sum(np.log(corr/sumage))
    
    loss/=num_train
    loss+= reg*np.sum(W*W)
    
    softmax = score/sumage
    
    softmax[range(num_train), y] -= 1
    
    dW = X.T.dot(softmax)
    
    dW /= num_train
    dW+= reg*2*W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
