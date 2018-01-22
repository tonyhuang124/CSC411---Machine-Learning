import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

BATCHES = 50
j = 0 #for Q3.6 j selection

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''

    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else: 
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch



def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)



def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist[0][0]



# TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    X = np.matrix(X)
    y = np.matrix(y).transpose()
    w = np.matrix(w).transpose()
    del_L =(2*np.dot(np.dot(X.transpose(),X),w)-2*np.dot(X.transpose(),y))/X.shape[0]

    return del_L

def compute_variance(A):
    A = np.array(A)
    mean = A.mean()
    semi_result = 0.0
    for x in A:
       semi_result = (x - mean)**2 +semi_result

    result = semi_result / len(A)

    return result
def main():
    # Load data and randomly initialise weightsS
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    # Example usage


    k= 500
    sum_var = np.zeros([13,])
    for s in range(k):
        X_b, y_b = batch_sampler.get_batch()
        sum_var = lin_reg_gradient(X_b, y_b, w) + sum_var
    computed_gradient = sum_var/k
    true_gradient = lin_reg_gradient(X, y, w)
    cos_similarity = cosine_similarity(np.array(computed_gradient.transpose())[0],np.array(true_gradient.transpose())[0])
    l2_distance = l2(np.array(computed_gradient).transpose(),np.array(true_gradient).transpose())
    print("Cosine_similarity: {}".format(cos_similarity))
    print("Squared distance: {}".format(l2_distance))

    variance_list = []
    m_list = []
    plt.plot(true_gradient)
    plt.plot(computed_gradient)
    plt.show()
    for m in range(1,401):
        m_list.append(m)
        new_batch_sample =BatchSampler(X, y, m)
        j_list = []
        for s in range(k):
            XX_b, yy_b = new_batch_sample.get_batch()
            j_list.append(lin_reg_gradient(XX_b, yy_b, w)[0,0])
        variance_list.append(compute_variance(j_list))
    plt.plot(np.log(m_list),np.log(variance_list))
    plt.ylabel('Log_Variances')
    plt.xlabel('Log_m')
    plt.show()



if __name__ == '__main__':
    main()