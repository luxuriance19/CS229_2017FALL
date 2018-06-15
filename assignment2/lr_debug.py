from __future__ import division
import numpy as np

try:
    xrange
except NameError:
    xrange = range

def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X

def load_data(filename):
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y

def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad

def calc_loss(X, Y, theta):
    return np.mean(np.log(1+np.exp(-Y*X.dot(theta))))

def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta  - learning_rate * (grad)
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return
                   
def lr2(X,Y):
    m,n = X.shape
    theta = np.zeros(n)
    learning_rate = 10
    
    thetas = []
    losses = []
    
    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        
        if i % 10000 == 0:
            thetas.append(theta)
            loss = calc_loss(X, Y, theta)
            losses.append(loss)
            print('Fininshed %d iterations'%i)
            print('loss at %d iterations %f'%(i, loss))
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break  
        if i % 200000 == 0:
            break
    return thetas, losses
                   

def main():
    print('==== Training model on data set A ====')
    Xa, Ya = load_data('data_a.txt')
    logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = load_data('data_b.txt')
    logistic_regression(Xb, Yb)

    return

if __name__ == '__main__':
    main()
