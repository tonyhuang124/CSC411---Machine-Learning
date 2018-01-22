'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means

    for idx,label in enumerate(train_labels):
        means[int(label)] = means[int(label)] + np.array(train_data[idx])

    means = means/700.0
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    sorted_train_data = [[] for i in range(10)]
    # Compute covariances

    means = compute_mean_mles(train_data, train_labels)

    for idx, label in enumerate(train_labels):
        sorted_train_data[int(label)].append(train_data[idx])


    for i in range(len(sorted_train_data)):
        sorted_train_data[i] = np.matrix(sorted_train_data[i])
        for col in range(0,64):
            for another_col in range(0,64):
                covariances[i][another_col][col] = np.dot((sorted_train_data[i][:,col] - means[i][col]).transpose(),(sorted_train_data[i][:,another_col] - means[i][another_col]))/700.0

    for idx in range(0,10):
        covariances[idx] = covariances[idx] + np.identity(64)*0.01

    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    all_concat = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        log_cov_diag = np.log(cov_diag).reshape((8, 8))
        if all_concat == []:
            all_concat = log_cov_diag
        else:
            all_concat = np.concatenate((all_concat,log_cov_diag),axis=1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    ans = [];
    for idx in range(len(digits)):
        probality_list = []
        for i in range(0,10):
            exp_part=np.exp(np.array(np.dot(np.dot((-1 / 2) * (digits[idx] - means[i]).transpose(),np.linalg.inv(np.matrix(covariances[i]))),(digits[idx] - means[i])))[0][0])
            temp_ans = np.log(((2*np.pi)**(-64/2))*(np.linalg.det(covariances[i])**(-1/2))*exp_part)
            probality_list.append(temp_ans)
        ans.append(probality_list)
    ans = np.array(ans)
    return ans

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    P_y = 1/10
    P_x_y = np.exp(generative_likelihood(digits, means, covariances))
    P_y_x = np.zeros(P_x_y.shape)
    for i in range(len(P_x_y)):
        P_x = np.sum(P_x_y[i])*P_y
        for class_label in range(0,10):
            P_y_x[i][class_label] = (P_x_y[i][class_label]*P_y)/P_x
    ans = np.log(P_y_x)
    return ans

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    ans = 0.0
    for idx,label in enumerate(labels):
        ans = ans + cond_likelihood[idx][int(label)]
    return ans/len(labels)

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    ans = []
    for idx in range(len(cond_likelihood)):
        ans.append(np.argmax(cond_likelihood[idx]))

    return np.array(ans)

def classification_accuracy(digits,digits_labels,means,covariances):
    predict_labels = classify_data(digits,means,covariances)
    correct_rate = 0;
    for idx in range(len(digits)):
        if predict_labels[idx] == digits_labels[idx]:
            correct_rate = correct_rate + 1

    return correct_rate/len(digits_labels)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    plot_cov_diagonal(covariances)
    # 2
    print("average conditional log-likelihood for Training data: ",avg_conditional_likelihood(train_data,train_labels, means, covariances))
    print("average conditional log-likelihood for Test data: ",avg_conditional_likelihood(test_data, test_labels, means, covariances))
    # 3
    print("accuracy on the train data: ", classification_accuracy(train_data,train_labels,means,covariances))
    print("accuracy on the test data: ", classification_accuracy(test_data, test_labels, means, covariances))


    # Evaluation

if __name__ == '__main__':
    main()