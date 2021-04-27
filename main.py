import csv
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import time



def plot_adaboost(X: np.ndarray,
                  y: np.ndarray,
                  clf=None,
                  sample_weights: Optional[np.ndarray] = None,
                  annotate: bool = False,
                  ax: Optional[mpl.axes.Axes] = None) -> None:
    """ Plot ± samples in 2D, optionally with decision boundary """

    assert set(y) == {-1, 1}, 'Expecting response labels to be ±1'

    if not ax:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        fig.set_facecolor('white')

    pad = 1
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    if sample_weights is not None:
        sizes = np.array(sample_weights) * X.shape[0] * 100
    else:
        sizes = np.ones(shape=X.shape[0]) * 100

    X_pos = X[y == 1]
    sizes_pos = sizes[y == 1]
    ax.scatter(*X_pos.T, s=sizes_pos, marker='+', color='red')

    X_neg = X[y == -1]
    sizes_neg = sizes[y == -1]
    ax.scatter(*X_neg.T, s=sizes_neg, marker='.', c='blue')

    if clf:
        plot_step = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # If all predictions are positive class, adjust color map acordingly
        if list(np.unique(Z)) == [1]:
            fill_colors = ['r']
        else:
            fill_colors = ['b', 'r']

        ax.contourf(xx, yy, Z, colors=fill_colors, alpha=0.2)

    if annotate:
        for i, (x, y) in enumerate(X):
            offset = 0.05
            ax.annotate(f'$x_{i + 1}$', (x + offset, y - offset))

    ax.set_xlim(x_min + 0.5, x_max - 0.5)
    ax.set_ylim(y_min + 0.5, y_max - 0.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')



class AdaBoost:
    """ AdaBoost enemble classifier from scratch """

    def __init__(self):
        self.stumps = None
        self.stump_weights = None
        self.errors = None
        self.sample_weights = None

    def _check_X_y(self, X, y):
        """ Validate assumptions about format of input data"""
        assert set(y) == {-1, 1}, 'Response variable must be ±1'
        return X, y

    def predict(self, X):
        """ Make predictions using already fitted model """
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        return np.sign(np.dot(self.stump_weights, stump_preds))

    def fit(self, X: np.ndarray, y: np.ndarray, iters: int):
        """ Fit the model using training data """

        X, y = self._check_X_y(X, y)
        n = X.shape[0]

        # init numpy arrays
        self.sample_weights = np.zeros(shape=(iters, n))
        self.stumps = np.zeros(shape=iters, dtype=object)
        self.stump_weights = np.zeros(shape=iters)
        self.errors = np.zeros(shape=iters)

        # initialize weights uniformly
        self.sample_weights[0] = np.ones(shape=n) / n

        for t in range(iters):
            # fit  weak learner
            curr_sample_weights = self.sample_weights[t]
            stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            stump = stump.fit(X, y, sample_weight=curr_sample_weights)

            # calculate error and stump weight from weak learner prediction
            stump_pred = stump.predict(X)
            err = curr_sample_weights[(stump_pred != y)].sum()  # / n
            stump_weight = np.log((1 - err) / err) / 2

            # update sample weights
            new_sample_weights = (
                    curr_sample_weights * np.exp(-stump_weight * y * stump_pred)
            )

            new_sample_weights /= new_sample_weights.sum()

            # If not final iteration, update sample weights for t+1
            if t + 1 < iters:
                self.sample_weights[t + 1] = new_sample_weights

            # save results of iteration
            self.stumps[t] = stump
            self.stump_weights[t] = stump_weight
            self.errors[t] = err

        return self


def truncate_adaboost(clf, t: int):
    """ Truncate a fitted AdaBoost up to (and including) a particular iteration """
    assert t > 0, 't must be a positive integer'
    from copy import deepcopy
    new_clf = deepcopy(clf)
    new_clf.stumps = clf.stumps[:t]
    new_clf.stump_weights = clf.stump_weights[:t]
    return new_clf


def plot_staged_adaboost(X, y, clf, iters=10):
    """ Plot weak learner and cumulaive strong learner at each iteration. """

    # larger grid
    fig, axes = plt.subplots(figsize=(8, iters * 3),
                             nrows=5,
                             ncols=2,
                             sharex=True,
                             dpi=100)

    fig.set_facecolor('white')

    _ = fig.suptitle('Decision boundaries by iteration')
    for i in range(iters):
        ax1, ax2 = axes[i]

        # Plot weak learner
        _ = ax1.set_title(f'Weak learner at t={i + 1}')
        plot_adaboost(X, y, clf.stumps[iter_arr[i]],
                      sample_weights=clf.sample_weights[i],
                      annotate=False, ax=axes[i//2,i%2])

    time.sleep(1)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()





'''
def plot_staged_adaboost(X, y, clf, iters=10):
    """ Plot weak learner and cumulaive strong learner at each iteration. """

    # larger grid
    fig, axes = plt.subplots(figsize=(8, iters * 3),
                             nrows=iters,
                             ncols=2,
                             sharex=True,
                             dpi=100)

    fig.set_facecolor('white')

    _ = fig.suptitle('Decision boundaries by iteration')
    for i in range(iters):
        ax1, ax2 = axes[i]

        # Plot weak learner
        _ = ax1.set_title(f'Weak learner at t={i + 1}')
        plot_adaboost(X, y, clf.stumps[i],
                      sample_weights=clf.sample_weights[i],
                      annotate=False, ax=ax1)

        # Plot strong learner
        trunc_clf = truncate_adaboost(clf, t=i + 1)
        _ = ax2.set_title(f'Strong learner at t={i + 1}')
        plot_adaboost(X, y, trunc_clf,
                      sample_weights=clf.sample_weights[i],
                      annotate=False, ax=ax2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
filename1 = 'chips.csv'


df = pd.read_csv(filename1, na_values="?")



X = df.values[:, :(df.shape[1] - 1)]
Y = df.values[:, (df.shape[1] - 1)]
# print(X_train, '====', Y)
Nlbl = len(np.unique(Y))  # Number of classes

X_norm = (X - X.min(0)) / X.ptp(0)  # X.ptp - peak to peak range
Y_norm = (Y - Y.min(0)) / Y.ptp(0)
'''

from sklearn.datasets import make_gaussian_quantiles
#from sklearn.model_selection import train_test_split




######################
file=csv.reader(open('chips.csv','r'))
#file=csv.reader(open('geyser.csv','r'))
next(file, None)  # skip the headers
x1=[]
x2=[]
y =[]
for row in file:
    x1.append(float(row[0]))
    x2.append(float(row[1]))
    y.append(1) if row[2] in ("P") else y.append(0)

X_train = np.c_[x1, x2]
X = np.asarray(X_train)
y = np.asarray(y)
y = (y*2)-1

####################

#clf = AdaBoost().fit(X, y, iters=61)
iter_arr = [1, 2, 3, 5, 8, 13, 21, 34, 55]
for i in range(8):

    clf = AdaBoost().fit(X, y, iters=iter_arr[i])
    train_err = (clf.predict(X) != y).mean()
    print('Train error for ' + str(iter_arr[i]) + f' iterations: {train_err:.1%}')

    plot_adaboost(X, y, clf)

    plt.title(iter_arr[i])

#clf = AdaBoost().fit(X, y, iters=47)
clf = AdaBoost().fit(X, y, iters=55)
train_err = (clf.predict(X) != y).mean()
print('Train error for ' + str(55) + f' iterations: {train_err:.1%}')
plot_adaboost(X, y, clf)
plt.title(55)

plt.show()

#plot_staged_adaboost(X, y, clf)



###################
'''
clf = AdaBoost().fit(X, y, iters=50)
plot_adaboost(X, y, clf)

train_err = (clf.predict(X) != y).mean()
print(f'Train error: {train_err:.1%}')

########################

from sklearn.ensemble import AdaBoostClassifier

bench = AdaBoostClassifier(n_estimators=50, algorithm='SAMME').fit(X, y)
plot_adaboost(X, y, bench)

train_err = (bench.predict(X) != y).mean()
print(f'Train error: {train_err:.1%}')


plot_adaboost(X,y)
'''