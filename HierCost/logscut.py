'''TODO: Implement Logistic Multi-label classifier'''

import numpy as np
import time
import scipy.sparse
import scipy
from logbase import LogisticBase
from logcost import LogisticCost
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

class LogisticScut(LogisticBase):

    def __init__(self, *args,**kwargs):
        super(LogisticScut, self).__init__(*args,**kwargs)
        self.threshold = 0

    def scut_threshold(self, scores, y_true):
        sorted_scores = np.sort(scores)[::-1]
        midpoints = (sorted_scores[1:] + sorted_scores[:-1])/2
        best_thresh, best_f1 = sorted_scores[0], 0
        for threshold in midpoints:
            y_pred = np.array(scores > threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_thresh = threshold
                best_f1 = f1
        return best_thresh

    def fit(self, X, y, cost_vector = []):
        '''Train the model'''

        # if cost vector not provided the default cost is 1 for all examples
        num_examples  = X.shape[0]
        if len(cost_vector) == 0:
            cost_vector = np.ones((num_examples,1))
        else:
            cost_vector = np.array(cost_vector).reshape((num_examples,1))

        # split dataset first, then tune the threshold
        if np.sum(y==1) < 2:
            self.threshold = 0
        else:
            niter = 1
            dev_thresholds = []
            sss = StratifiedShuffleSplit(y, n_iter=niter, test_size=0.3)
            for train_index, dev_index in sss:
                X_train, X_dev = X[train_index,:], X[dev_index,:]
                y_train, y_dev = y[train_index], y[dev_index]
                dev_base_model = LogisticCost(rho=self.rho, intercept_scaling=self.intercept_scaling)
                dev_base_model.fit(X_train, y_train, cost_vector[train_index])
                dev_scores = dev_base_model.decision_function(X_dev)
                dev_thresholds.append(self.scut_threshold(dev_scores, y_dev))
            self.threshold = np.mean(dev_thresholds)

        base_model = LogisticCost(rho=self.rho, intercept_scaling=self.intercept_scaling)
        base_model.fit(X, y, cost_vector)
        self.base_model = base_model


    def predict(self, X):
        '''predict the labels of each instance'''

        scores = self.base_model.decision_function(X)
        y_pred = np.array(scores > self.threshold).astype(int)
        return y_pred.flatten()

