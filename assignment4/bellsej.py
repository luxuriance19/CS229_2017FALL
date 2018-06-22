### Independent Components Analysis
###
### This program requires a working installation of:
###
### On Mac:
###     1. portaudio: On Mac: brew install portaudio
###     2. sounddevice: pip install sounddevice
###
### On windows:
###      pip install pyaudio sounddevice
###

import sounddevice as sd
import numpy as np

Fs = 11025

def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('mix.dat')
    return mix

def play(vec):
    sd.play(vec, Fs, blocking=True)
    
# 这个算法当中,首先要选择初始化的源音频的数据作为密度特征的数据
# 这里选择sigmoid函数做为CDF
def sigmoid(x):
    
	return 1. / (1 + np.exp(-x))
	
def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    ######## Your code here ##########
    for alpha in anneal:
		# 这里对W, X做矩阵运算
        for i in range(M):
            xi = X[i,:]
            # print(W.dot(xi))
            g = sigmoid(W.dot(xi))
            delta2 = np.linalg.inv(W.T)
            delta1 = (1 - 2 * g).reshape(-1,1).dot(xi[np.newaxis,:])
            # print(delta1)
            W = W + alpha * (delta1 + delta2) 
    ###################################
    return W

def unmix(X, W):
    S = np.zeros(X.shape)

    ######### Your code here ##########
    S = X.dot(W.T)
    ##################################
    return S

def main():
    X = normalize(load_data())

    for i in range(X.shape[1]):
        print('Playing mixed track %d' % i)
        play(X[:, i])

    W = unmixer(X)
    S = normalize(unmix(X, W))

    for i in range(S.shape[1]):
        print('Playing separated track %d' % i)
        play(S[:, i])

if __name__ == '__main__':
    main()
