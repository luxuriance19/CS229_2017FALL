import numpy as np

def readMatrix(file):
    '''
    readMatrix: return matrix shape is (emails_num, words_num)
    each row represent a unique email, j-th column represents times of the j-th token appeared in the specific email.
    
    tokens: list of words
    
    np.array(y) the emails classification sign, spam emails are indicated as 1, non_spam: 0
    '''	
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    '''
    return log probability of the prior and conditional
    '''
    
    
    state1 = {}
    # N = matrix.shape[1]
    num_emails, N = matrix.shape
    #print(N)
    ###################
    spam = matrix[category == 1, :]
    non_spam = matrix[category == 0, :]
    
    non_spam_words_num = np.sum(non_spam)
    spam_words_num = np.sum(spam)
    #print(non_spam_words_num)
    #print(spam_words_num)
    
    state1['non_spam_log_prior'] = np.log(non_spam.shape[0] * 1.0 / num_emails)
    state1['spam_log_prior'] = np.log(spam.shape[0] * 1.0 / num_emails)
    
    state1['phi_log_spam'] = np.log((spam.sum(axis = 0) + 1) * 1.0 / (spam_words_num + N))
    state1['phi_log_non_spam'] = np.log((non_spam.sum(axis = 0) + 1) * 1.0 / (non_spam_words_num + N))
   
    ###################
    return state1

def nb_test(matrix, state):
    '''
    calculate the posterior probability
    the posterior probability's denominator is the same p(x), so we just need compare numerator
    we choose np.exp(log_prob), then the numerator is np.exp(conditional log(p(x|y)) + log(prior p(x)))
    so we just need compare log(p(x|y)) + log(p(x))
    p(x|y) = IIp(x_i|y)
    '''
    output = np.zeros(matrix.shape[0])
    ###################
    posterior_spam = np.sum(state['phi_log_spam'] * matrix, axis = 1) + state['spam_log_prior']
    posterior_non_spam = np.sum(state['phi_log_non_spam'] * matrix, axis = 1) + state['non_spam_log_prior']
    
    output[posterior_spam > posterior_non_spam] = 1
        
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)
    return error

def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
    return

if __name__ == '__main__':
    main()
