import numpy as np
import utils.metrics as metrics

class DecisionStump:
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        raise NotImplementedError("Subclasses must implement fit(self, X, y)")

    def predict(self, X):
        n, d = X.shape

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n)

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] > self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat
    
class ClassificationStumpErrorRate(DecisionStump):
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to leq to
                t = X[i, j]

                # Split the classes based on this threshold
                is_yes = X[:, j] > t
                y_yes_mode = metrics.mode(y[is_yes])
                y_no_mode = metrics.mode(y[~is_yes])  # ~ is "logical not"

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[~is_yes] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode


class ClassificationStumpInfoGain(DecisionStump):
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return
        
        minEntropy = metrics.entropy(count / n)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to leq to
                t = X[i, j]

                # Split the classes based on this threshold
                is_yes = X[:, j] > t

                y_yes = y[is_yes]
                y_no = y[~is_yes]

                y_yes_mode = metrics.mode(y_yes)
                y_no_mode = metrics.mode(y_no)  # ~ is "logical not"

                # Compute entropy
                entropy_yes = metrics.entropy(np.bincount(y_yes)/y_yes.size)
                entropy_no = metrics.entropy(np.bincount(y_no)/y_no.size)

                entropy_curr = (y_yes.size * entropy_yes + y_no.size * entropy_no)/y.size

                # Compare to minimum entropy so far
                if entropy_curr < minEntropy:
                    # This is the lowest entropy, store this value
                    minEntropy = entropy_curr
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode

class RegressionStump(DecisionStump):
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape

        y_mean = metrics.mean(y)

        self.y_hat_yes = y_mean
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return
        
        minSSE = metrics.sse(y)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to leq to
                t = X[i, j]

                # Split the classes based on this threshold
                is_yes = X[:, j] > t

                y_yes = y[is_yes]
                y_no = y[~is_yes]

                y_yes_mean = metrics.mean(y_yes)
                y_no_mean = metrics.mean(y_no)  # ~ is "logical not"

                # Compute sse
                sse_curr = metrics.sse(y_yes) + metrics.sse(y_no)

                # Compare to minimum sse so far
                if sse_curr < minSSE:
                    # This is the lowest sse, store this value
                    minSSE = sse_curr
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mean
                    self.y_hat_no = y_no_mean

class DecisionTree:
    stump_model = None
    submodel_yes = None
    submodel_no = None

    def __init__(self, max_depth, stump_class=ClassificationStumpInfoGain):
        self.max_depth = max_depth
        self.stump_class = stump_class

    def fit(self, X, y):
        # Fits a decision tree using greedy recursive splitting

        # Learn a decision stump
        stump_model = self.stump_class()
        stump_model.fit(X, y)

        if self.max_depth <= 1 or stump_model.j_best is None:
            # If we have reached the maximum depth or the decision stump does
            # nothing, use the decision stump

            self.stump_model = stump_model
            self.submodel_yes = None
            self.submodel_no = None
            return

        # Fit a decision tree to each split, decreasing maximum depth by 1
        j = stump_model.j_best
        value = stump_model.t_best

        # Find indices of examples in each split
        yes = X[:, j] > value
        no = X[:, j] <= value

        # Fit decision tree to each split
        self.stump_model = stump_model
        self.submodel_yes = DecisionTree(
            self.max_depth - 1, stump_class=self.stump_class
        )
        self.submodel_yes.fit(X[yes], y[yes])
        self.submodel_no = DecisionTree(
            self.max_depth - 1, stump_class=self.stump_class
        )
        self.submodel_no.fit(X[no], y[no])

    def predict(self, X):
        n, d = X.shape
        y = np.zeros(n)

        # GET VALUES FROM MODEL
        j_best = self.stump_model.j_best
        t_best = self.stump_model.t_best
        y_hat_yes = self.stump_model.y_hat_yes

        if j_best is None:
            # If no further splitting, return the best label
            y = y_hat_yes * np.ones(n)

        # the case with depth=1, just a single stump.
        elif self.submodel_yes is None:
            return self.stump_model.predict(X)

        else:
            # Recurse on both sub-models
            j = j_best
            value = t_best

            yes = X[:, j] > value
            no = X[:, j] <= value

            y[yes] = self.submodel_yes.predict(X[yes])
            y[no] = self.submodel_no.predict(X[no])

        return y