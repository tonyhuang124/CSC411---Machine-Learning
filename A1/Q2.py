# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B,tau):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = -(A_norm + B_norm - 2 * A.dot(B.transpose()))/(2*(tau**2))
    return dist


# helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses

def chunk(list, idx):
    #chunk the list/array into idx parts
    splited_avg = len(list) / float(idx)
    end = 0.0
    result = []
    while end < len(list):
        result.append(list[int(end):int(end + splited_avg)])
        end = end + splited_avg
    return result


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    dist = l2(test_datum.transpose() ,x_train,tau)
    base = np.sum(np.exp(dist))
    a = []
    for x in dist:
        a.append(np.exp(x)/base)
    a = np.array(a)[0]
    A = np.diag(a)
    w = np.dot(np.dot(np.dot(np.linalg.solve(np.dot(np.dot(x_train.transpose(),A),x_train) + lam*np.identity(14),np.identity(14)),x_train.transpose()),A),y_train)
    y_hat = np.dot(w,test_datum)[0]
    return y_hat



def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO
    losses = []
    new_idx = chunk(idx,k)
    for s in range(len(new_idx)):
        train_data = []
        train_target = []
        test_data = []
        test_target = []
        for item in idx:
            if item in new_idx[s]:
                test_data.append(x[item])
                test_target.append(y[item])
            else:
                train_data.append(x[item])
                train_target.append(y[item])
        train_data = np.array(train_data)
        train_target = np.array(train_target)
        test_target = np.array(test_target)
        test_data = np.array(test_data)
        losses.append(run_on_fold(test_data, test_target, train_data,train_target, taus))


    return np.array(losses)



if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    losses = run_k_fold(x, y, taus, k=5)
    new_losses = np.array(np.matrix(losses).mean(0))[0]
    plt.plot(taus, new_losses)
    plt.ylabel('losses')
    plt.xlabel('taus')
    plt.show()
    print("min loss = {}".format(new_losses.min()))
