# import sklearn for loading data and splitting data

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


# Load and Store the feature data
X = datasets.load_breast_cancer().data

# Load and store the target data
y = datasets.load_breast_cancer().target

# split the data using Scikit-Learn's train_test_split

#NOTE: I forgot to set a random state here, so the results may be slighlty different compared to Table 1
X_train, X_test, y_train, y_test = train_test_split(X, y)

# evaluate different components solutions based on different lambdas
# set SSE_ratio to an unreachable (e.g., -2)number so that extraction does not stop
# set max_comp = 10 so that only 10 components are extracted


## lambda = 0
# 10 iterations
components_0_10,SSEs_0_10 = optimal_component(X_train , max_iter = 10, max_comp = 10, SSE_ratio = -2, lam = 0)

# 20 iterations
components_0_20,SSEs_0_20 = optimal_component(X_train , max_iter = 20, max_comp = 10, SSE_ratio = -2, lam = 0)


## lambda = .1
# 10 iterations
components_01_10,SSEs_01_10 = optimal_component(X_train , max_iter = 10, max_comp = 10, SSE_ratio = -2, lam = .1)

# 20 iterations
components_01_20,SSEs_01_20 = optimal_component(X_train , max_iter = 20, max_comp = 10, SSE_ratio = -2, lam = .1)


## lambda = .5
# 10 iterations
components_05_10,SSEs_05_10 = optimal_component(X_train , max_iter = 10, max_comp = 10, SSE_ratio = -2, lam = .5)

# 20 iterations
components_05_20,SSEs_05_20 = optimal_component(X_train , max_iter = 20, max_comp = 10, SSE_ratio = -2, lam = .5)


## lambda = 1
# 10 iterations
components_1_10,SSEs_1_10 = optimal_component(X_train , max_iter = 10, max_comp = 10, SSE_ratio = -2, lam = 1)

# 20 iterations
components_1_20,SSEs_1_20 = optimal_component(X_train , max_iter = 20, max_comp = 10, SSE_ratio = -2, lam = 1)

# get results in array (mimicing Table 1)
Table_1 = np.array([SSEs_0_10,
                    SSEs_0_20,
                    SSEs_01_10,
                    SSEs_01_20,
                    SSEs_05_10,
                    SSEs_05_20,
                    SSEs_1_10,
                    SSEs_1_20]).T
                 
# The SSEs object now contains results in Table 1 (slight variation due to different random state)

print(Table_1)
