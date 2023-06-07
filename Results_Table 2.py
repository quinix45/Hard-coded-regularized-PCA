import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Load and Store the feature data
X = datasets.load_breast_cancer().data

# Load and store the target data
y = datasets.load_breast_cancer().target


# data used for logistic classifier training
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1997)


# Data for component learning (3 differnt split conditions in Table 2) 

# 80%
X_train_80, X_test_80, y_train_80, y_test_80 = train_test_split(X, y, 
                                                                test_size=.20, 
                                                                train_size=.80, 
                                                                random_state = 80)

# 60%
X_train_60, X_test_60, y_train_60, y_test_60 = train_test_split(X, y, 
                                                                test_size=.40, 
                                                                train_size=.60, 
                                                                random_state = 60)

# 40%
X_train_40, X_test_40, y_train_40, y_test_40 = train_test_split(X, y, 
                                                                test_size=.60, 
                                                                train_size=.40,
                                                                random_state = 40)
                                                                

# find 6 components for each lambda X split condition in Table 2

# lambda = 0

components_l0_80,SSEs_l0_80 = optimal_component(X_train_80 , max_iter = 10, max_comp = 6, SSE_ratio = -2, lam = 0)
components_l0_60,SSEs_l0_60 = optimal_component(X_train_60 , max_iter = 10, max_comp = 6, SSE_ratio = -2, lam = 0)
components_l0_40,SSEs_l0_40 = optimal_component(X_train_40 , max_iter = 10, max_comp = 6, SSE_ratio = -2, lam = 0)



# lambda .5

components_l05_80,SSEs_l05_80 = optimal_component(X_train_80 , max_iter = 10, max_comp = 6, SSE_ratio = -2, lam = .5)
components_l05_60,SSEs_l05_60 = optimal_component(X_train_60 , max_iter = 10, max_comp = 6, SSE_ratio = -2, lam = .5)
components_l05_40,SSEs_l05_40 = optimal_component(X_train_40 , max_iter = 10, max_comp = 6, SSE_ratio = -2, lam = .5)



# lambda 1

components_l1_80,SSEs_l1_80 = optimal_component(X_train_80 , max_iter = 10, max_comp = 6, SSE_ratio = -2, lam = 1)
components_l1_60,SSEs_l1_60 = optimal_component(X_train_60 , max_iter = 10, max_comp = 6, SSE_ratio = -2, lam = 1)
components_l1_40,SSEs_l1_40 = optimal_component(X_train_40 , max_iter = 10, max_comp = 6, SSE_ratio = -2, lam = 1)



# Create training and test detasets of 6 reconstructed feature for each condition (9 train, 9 test)
# The comp_feature() function is used to facilitate this process

# training data

## train data Lambda 0

dat_PCA_l0_80_train = comp_feature(standardizer(X_train), components_l0_80, features = 6)
dat_PCA_l0_60_train = comp_feature(standardizer(X_train), components_l0_60, features = 6)
dat_PCA_l0_40_train = comp_feature(standardizer(X_train), components_l0_40, features = 6)

## train data Lambda .5

dat_PCA_l05_80_train = comp_feature(standardizer(X_train), components_l05_80, features = 6)
dat_PCA_l05_60_train = comp_feature(standardizer(X_train), components_l05_60, features = 6)
dat_PCA_l05_40_train = comp_feature(standardizer(X_train), components_l05_40, features = 6)


## train data Lambda 1

dat_PCA_l1_80_train = comp_feature(standardizer(X_train), components_l1_80, features = 6)
dat_PCA_l1_60_train = comp_feature(standardizer(X_train), components_l1_60, features = 6)
dat_PCA_l1_40_train = comp_feature(standardizer(X_train), components_l1_40, features = 6)

# test data

## test data Lambda 0

dat_PCA_l0_80_test = comp_feature(standardizer(X_test), components_l0_80, features = 6)
dat_PCA_l0_60_test = comp_feature(standardizer(X_test), components_l0_60, features = 6)
dat_PCA_l0_40_test = comp_feature(standardizer(X_test), components_l0_40, features = 6)

## test data Lambda .5

dat_PCA_l05_80_test = comp_feature(standardizer(X_test), components_l05_80, features = 6)
dat_PCA_l05_60_test = comp_feature(standardizer(X_test), components_l05_60, features = 6)
dat_PCA_l05_40_test = comp_feature(standardizer(X_test), components_l05_40, features = 6)


## test data Lambda 1


dat_PCA_l1_80_test = comp_feature(standardizer(X_test), components_l1_80, features = 6)
dat_PCA_l1_60_test = comp_feature(standardizer(X_test), components_l1_60, features = 6)
dat_PCA_l1_40_test = comp_feature(standardizer(X_test), components_l1_40, features = 6)



# create list of training and test arrays to loop over to get Tabel 2 accuracy results

## train list
dat_train_list = [dat_PCA_l0_80_train, 
                  dat_PCA_l0_60_train, 
                  dat_PCA_l0_40_train, 
                  dat_PCA_l05_80_train, 
                  dat_PCA_l05_60_train, 
                  dat_PCA_l05_40_train,
                  dat_PCA_l1_80_train, 
                  dat_PCA_l1_60_train, 
                  dat_PCA_l1_40_train]

## test list
dat_test_list =  [dat_PCA_l0_80_test, 
                  dat_PCA_l0_60_test, 
                  dat_PCA_l0_40_test, 
                  dat_PCA_l05_80_test, 
                  dat_PCA_l05_60_test, 
                  dat_PCA_l05_40_test,
                  dat_PCA_l1_80_test, 
                  dat_PCA_l1_60_test, 
                  dat_PCA_l1_40_test]
                  
# First clculate accuracy for 30 features

logisticRegr = LogisticRegression()
fit = logisticRegr.fit(standardizer(X_train), y_train)

score2 = fit.score(standardizer(X_test), y_test)
# accuracy = 0.9790209790209791


# Now create appary of accuracies 

# arrays of 0 with dimensions of Table 2
Table_2 = np.zeros((dat_PCA_l0_40_test.shape[1],len(dat_train_list)))

# Loop to run logistic classifier for all cells in table 2
for i in range(Table_2.shape[1]):
  for j in range(Table_2.shape[0]): 
    logisticRegr = LogisticRegression()
    fit = logisticRegr.fit(dat_train_list[i][:,0:j+1], y_train)
    score = fit.score(dat_test_list[i][:,0:j+1], y_test)
    Table_2[j,i] = score2 - score

# the `Table_2` object now contains results in Table 2

print(Table_2)                 
                                                                
