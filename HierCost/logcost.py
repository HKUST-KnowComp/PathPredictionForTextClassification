'''Cost-sensitive binary logistic regression.
If the cost vector is 1 then it should behave like standard logistic regression'''
# NOTE: generates unbiased linear classifiers
# TODO: This method works for multi-class OVR classifier but the cost_vector cosideration
# as implemented right now takes the same cost for all OVR classifiers.

import numpy as np
import time
import scipy.sparse
import scipy
from logbase import LogisticBase

class LogisticCost(LogisticBase):

    def __init__(self, *args,**kwargs):
        super(LogisticCost, self).__init__(*args,**kwargs)


    def fit(self, X, y, cost_vector = []):
        '''Train the model'''

        # if cost vector not provided the default cost is 1 for all examples
        num_examples  = X.shape[0]
        if len(cost_vector) == 0:
            self.cost_vector = np.ones((num_examples,1))
        else:
            self.cost_vector = np.array(cost_vector).reshape((num_examples,1))

        if self.solver == 'agm':
            super(LogisticCost, self).optimize_objective_agm(X, y)
        else:
            super(LogisticCost, self).optimize_objective_lbfgs(X, y)

        del self.cost_vector

    def _function_value(self, W, X, Y):
        '''Compute objective function value'''
        regularizer = np.linalg.norm(W)**2
        loss_vector = self._log_loss(X, Y, W)
        np.multiply(self.cost_vector, loss_vector, loss_vector)
        loss = np.sum(loss_vector)
        value = loss + self.rho*regularizer
        return value

    def _gradient(self, W, X, Y):
        W.shape = (X.shape[1], Y.shape[1])
        grad_loss = -Y/(1+np.exp(Y*(X.dot(W))))
        grad_loss = np.multiply(grad_loss, self.cost_vector)
        grad_loss = X.T.dot(grad_loss) # TODO: this is CSR improper multiplication
        gradient = grad_loss + 2*self.rho*W
        if self.solver == 'lbfgs':
            W.shape = (X.shape[1]*Y.shape[1])
            return gradient.flatten()
        else:
            return gradient
