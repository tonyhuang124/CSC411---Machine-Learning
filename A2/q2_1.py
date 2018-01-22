'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
from sklearn.model_selection import KFold


# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        digit = -1
        digit_list = []
        dist_list = self.l2_distance(test_point)

        if k<=0:
            print("K must greater than 0!")
        elif k==1:
            index = dist_list.argsort()[:1][0]
            digit = self.train_labels[index]
        else:
            index_list = dist_list.argsort()[:k]
            for item in index_list:
                digit_list.append(self.train_labels[item])
            count_list = []
            for label in range(0,10):
                count_list.append(digit_list.count(label))
            digit_count = max(count_list);
            digit = count_list.index(max(count_list))
            count_list[digit] = 0;
            second_large_digit_count = max(count_list);
            second_large_digit = count_list.index(max(count_list))
            if digit_count == second_large_digit_count:
                idx = 0
                while(digit_list[idx] != digit and digit_list[idx] != second_large_digit ):
                    idx = idx + 1
                if digit != digit_list[idx]:
                    digit = second_large_digit
        return digit

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    k_score_list = []
    for k in k_range:
        kf = KFold(n_splits=10)
        cv_list = []
        for train_index, test_index in kf.split(train_data):
            X_train, X_test = train_data[train_index], train_labels[train_index]
            y_train, y_test = train_data[test_index], train_labels[test_index]
            knn = KNearestNeighbor(X_train,X_test)
            cv_list.append(classification_accuracy(knn,k,y_train,y_test))
        k_score_list.append(np.array(cv_list).mean())
        print("Average accuracy across folds for K = ",k,": ",np.array(cv_list).mean())


    ans = k_range[np.argmax(np.array(k_score_list))]
    print("value of K: ",ans)
    return ans

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    incorrect_num = 0
    for i in range(len(eval_data)):
        predicted_label = knn.query_knn(eval_data[i], k)
        if predicted_label == eval_labels[i]:
            incorrect_num = incorrect_num +1
    correct_rate = incorrect_num/len(eval_labels)

    return correct_rate

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)
    # 2.1
    print("Training classification accuracy for K = 1:", classification_accuracy(knn, 1, train_data, train_labels))
    print("Test classification accuracy for K = 1:",classification_accuracy(knn, 1, test_data, test_labels))
    print("Training classification accuracy for K = 15:", classification_accuracy(knn, 15, train_data, train_labels))
    print("Test classification accuracy for K = 15:",classification_accuracy(knn, 15, test_data, test_labels))
    # Example usage:
    best_k = cross_validation(train_data, train_labels, k_range=np.arange(1, 16))
    print("Training classification accuracy for K =",best_k,":", classification_accuracy(knn, best_k, train_data, train_labels))
    print("Test classification accuracy for K =",best_k,":", classification_accuracy(knn, best_k, test_data, test_labels))
if __name__ == '__main__':
    main()