'''Base class for LogisticRegression type methods'''
import numpy as np
import time
import scipy.sparse
import scipy
import pandas as pd
import warnings
from sklearn import preprocessing

class LogisticBase(object):

    def __init__(self, rho, accelerated=True, max_iter = 2000, tol = 10**-4, beta=0.5,
            fit_intercept = True, intercept_scaling = 1, feature_scaling = 1, verbosity=0, solver='lbfgs', var_init_type=1):

        # initialized member variables
        self.rho = float(rho)
        self.accelerated = accelerated
        self.max_iter = max_iter
        self.tol = float(tol)
        self.beta = float(beta)
        self.is_sparse = False
        self.fit_intercept = fit_intercept
        self.intercept_scaling = float(intercept_scaling)
        self.solver = solver # agm, lbfgs
        self.var_init_type = var_init_type
        self.feature_scaling = feature_scaling

        # uninitialized member variables
        self.coef_ = None
        self.labels = None

        # member variables for debugging
        self.iteration_objectives = []
        self.verbosity = verbosity

    def _initialize_variables(self, shape):
        if self.var_init_type == 0:
            W = np.zeros(*shape)
            W_prev = np.zeros(*shape)
            warnings.warn("Initializing weights to zeros causes slow convergence")
        elif self.var_init_type == 1:
            W = np.random.randn(*shape)
            W_prev = np.random.randn(*shape)
            W[0] = W[0]/self.intercept_scaling
            W_prev[0] = W_prev[0]/self.intercept_scaling

        return W, W_prev

    def _check_data(self, y):
        class_labels = np.unique(y)
        num_tasks = len(class_labels)
        num_examples = y.shape[0]
        """
        if num_tasks == 1:
            raise ValueError("The number of classes has to be greater than one.")
        elif num_tasks == 2:
            if 1 in class_labels and -1 in class_labels:
                num_tasks = 1
                class_labels = np.array([-1, 1])
            elif 1 in class_labels and 0 in class_labels:
                num_tasks = 1
                class_labels = np.array([0, 1])
            else:
                raise ValueError("Unable to decide postive label")
        """
        if num_tasks == 2:
            if 1 in class_labels and -1 in class_labels:
                num_tasks = 1
                class_labels = np.array([-1, 1])
            elif 1 in class_labels and 0 in class_labels:
                num_tasks = 1
                class_labels = np.array([0, 1])

        lbin = preprocessing.LabelBinarizer(neg_label=-1, pos_label=1)
        lbin.fit(class_labels)
        y_bin = lbin.transform(y)
        return y_bin, class_labels, num_tasks


    def optimize_objective_lbfgs(self, X, y):
        '''Train the model'''


        X = self.append_unit_column(X)
        Y, self.labels, num_tasks = self._check_data(y)
        num_examples, num_features = X.shape

        W = self._initialize_variables((num_features,num_tasks))[0].flatten()
        W, fvalue, d  = scipy.optimize.fmin_l_bfgs_b(
                    self._function_value,
                    W,
                    fprime=self._gradient,
                    args = (X, Y))
        W.shape = (num_features, num_tasks)

        if self.fit_intercept:
            self.coef_ = W[1:,:]
            self.intercept_ = W[0,:]*self.intercept_scaling
        else:
            self.coef_ = W
            self.intercept_ = 0

        if hasattr(self, 'feature_scaling') and self.feature_scaling != 1:
            self.coef_ *= self.feature_scaling

        if num_tasks == 1:
            self.coef_ = self.coef_.flatten()

        if self.verbosity == 2:
            print(d)


    def optimize_objective_agm(self, X, y):
        '''Train the model'''

        X = self.append_unit_column(X)
        Y, self.labels, num_tasks = self._check_data(y)

        inv_step_size = 1
        n_term_check = 5
        W, W_prev = self._initialize_variables((X.shape[1], Y.shape[1]))

        iterations_info_df = pd.DataFrame(columns = ["objective", "grad_W_norm", "step_size", "step_time", "obj_S", "grad_S"])
        iterations_info_df.loc[0] = [self._function_value(W, X, Y), -1, 1, 0, 0, 0]

        for k in range(1,self.max_iter+1):

            iter_start_time = time.time()

            if self.accelerated:
                theta = (k-1)*1.0/(k+2)
                S = W + theta*(W-W_prev)
            else:
                S = W

            inv_step_size_temp = inv_step_size*(self.beta)
            grad_S = self._gradient(S, X, Y)
            func_value_S = self._function_value(S, X, Y)
            grad_S_norm = np.linalg.norm(grad_S)

            # convergence check
            # objective has not changed much in the last n_term_check iterations
            if k > n_term_check:
                if iterations_info_df.objective[-n_term_check:].std() < self.tol:
                    break

            # backtracking line search
            while True:

                diff_US = -1*grad_S/inv_step_size_temp
                C = S + diff_US

                try:
                    # apply proximal step if required
                    U = self._prox(C, 1.0/inv_step_size)
                except AttributeError:
                    U = C

                func_value_U = self._function_value(U, X, Y)
                quad_approx_U = func_value_S - 0.5/inv_step_size_temp*grad_S_norm**2

                if func_value_U <= quad_approx_U:
                    inv_step_size = inv_step_size_temp
                    W, W_prev = U, W
                    break
                else:
                    inv_step_size_temp = inv_step_size_temp/self.beta

            grad_W_norm = np.linalg.norm(self._gradient(W, X, Y)) if self.verbosity == 2 else -1
            iterations_info_df.loc[k] = [func_value_U, grad_W_norm, 1.0/inv_step_size, time.time() - iter_start_time, func_value_S, grad_S_norm]

        if self.fit_intercept:
            self.coef_ = W[1:,:]
            self.intercept_ = W[0,:]*self.intercept_scaling
        else:
            self.coef_ = W
            self.intercept_ = 0

        if hasattr(self, 'feature_scaling') and self.feature_scaling != 1:
            self.coef_ *= self.feature_scaling

        if num_tasks == 1:
            self.coef_ = self.coef_.flatten()

        if self.verbosity == 2:
            self._print_full_df(iterations_info_df.astype(float))


    def _print_full_df(self, dataframe):
        pd.set_option('display.max_rows', len(dataframe))
        print(dataframe)
        pd.reset_option('display.max_rows')

    def append_unit_column(self, X):
        '''Add unit column in the first position'''
        if hasattr(self,'feature_scaling'):
            X = self.feature_scaling*X

        if self.fit_intercept:
            unit_col = self.intercept_scaling*np.ones((X.shape[0], 1))
        newX = (scipy.sparse.hstack((unit_col, X))).tocsr()
        return newX

    def sparsify(self):
        self.coef_ = scipy.sparse.csc_matrix(self.coef_)
        self.is_sparse = True

    def decision_function(self, X):
        '''scores of each instance w.r.t. each class'''
        score = X.dot(self.coef_) + self.intercept_
        if isinstance(score, (scipy.sparse.csc_matrix, scipy.sparse.csr_matrix)):
            score = score.todense().A
        if len(self.labels) == 2:
            score = score.flatten()
        return score

    def predict(self, X):
        '''predict the labels of each instance'''
        scores = self.decision_function(X)
        if len(self.labels) == 2:
            # binary case
            y_index = (scores > 0).astype(int)
            y_pred = self.labels[y_index]
        else:
            # multiclass case
            y_index = np.argmax(scores, 1)
            y_pred = self.labels[y_index]

        return y_pred.flatten()

    def score(self, X, y):
        '''Returns the mean accuracy on the given test data and labels.'''
        y_pred = self.predict(X)
        y_arr = np.array(y)
        return np.mean(y_arr == y_pred)

    def predict_proba(self, X):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if len(prob.shape) == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob

    def _log_loss(self, X, Y, W):
        '''Computes logistic loss vector

            np.log(1+np.exp(-1*X.dot(W)*Y))

        direct computation of the above expression causes numerical error.
        for large positive values in -1*X.dot(W)*Y. Therefore we use
        the following expression.
        '''
        threshold = 50
        if self.solver == 'lbfgs':
            W.shape = (X.shape[1], Y.shape[1])
        loss_vector = -1*np.multiply(Y, X.dot(W))
        sel_index = (loss_vector < threshold)
        loss_vector[sel_index] = np.log1p(np.exp(loss_vector[sel_index]))
        if self.solver == 'lbfgs':
            W.shape = (X.shape[1]*Y.shape[1])
        return loss_vector
