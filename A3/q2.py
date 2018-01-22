import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

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

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self,lr,beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, x, y, c, w):
        # Update parameters using GD with momentum and return
        # the updated parameters
        N = np.size(x, 0)

        def func_grad(x_i, y_i, w):
            if y_i * (np.array((x_i*w))[0][0]) < 1:
                return -y_i * x_i
            else:
                return 0

        sum_value = 0
        for idx in range(N):
            grad_value = func_grad(x[idx], y[idx], w)
            sum_value = w + np.matrix(grad_value).transpose()

        w = w - self.lr * (c/N) * sum_value + self.beta * self.vel
        self.vel = - self.lr * (c/N) * sum_value + self.beta * self.vel
        return w




class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.matrix(np.random.normal(0.0, 0.1, feature_count)).transpose()

        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        ans = []
        for idx in range(np.size(X,0)):
            predict = np.array(X[idx]*self.w)[0][0]
            cost = max(0,1-y[idx]*predict)
            ans.append(cost)
        return np.array(ans)

    def grad(self, X, y,beta,iter,batchsize):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        N = np.size(X, 0)
        gradient = GDOptimizer(0.05,beta)
        original_w = self.w
        w_list = np.zeros((np.size(X, 1), 1))
        for t in range(0, iter):
            mini_batch = BatchSampler(X, y, batchsize)
            print('Training progress: ', (t/iter)*batchsize,'%')
        #     last_w = self.w
        #     last_loss = sum(self.hinge_loss(X, y))
        #     self.w = gradient.update_params(x_batch, y_batch, self.c, self.w)
        #     new_loss = sum(self.hinge_loss(X, y))
        #
        #     while new_loss < last_loss:
        #         last_loss = new_loss
        #         last_w = self.w
        #         self.w = gradient.update_params(x_batch, y_batch, self.c, self.w)
        #         new_loss = sum(self.hinge_loss(X, y))
        #
        #     self.w = last_w
        #     w_list = np.add(w_list, self.w)
        #     self.w = np.matrix(np.random.normal(0.0, 0.1, np.size(X, 1))).transpose()
        # self.w = w_list/500.0
            for step in range(0, N//batchsize):
                x_batch, y_batch = mini_batch.get_batch()
                self.w = gradient.update_params(x_batch, y_batch, self.c, self.w)
        return self.w

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        ans = []
        for idx in range(np.size(X,0)):
            prediction = np.array(X[idx]*self.w)[0][0]
            if prediction <= 0:
                ans.append(-1.0)
            else:
                ans.append(+1.0)
        return np.array(ans)

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(beta, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    alpha = 1.0
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]
    grad = 0
    for x in range(steps):
        # Optimize and update the history
        grad = - alpha * func_grad(w) + beta * grad
        w = w + grad
        w_history.append(w)
    return w_history

def optimize_svm(train_data, train_targets, C, beta, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''
    #train SVM
    svm_model = SVM(C,np.size(train_data, 1))
    svm_model.grad(train_data, train_targets,beta,500,100)
    return svm_model


if __name__ == '__main__':
    # a = optimize_test_function(0.0,10.0,200)
    # plt.plot(a,title = 'Beta = 0.0')
    #b = optimize_test_function(0.0, 10.0, 200)
    #plt.plot(b)
    #plt.title('beta = 0.0')
    #plt.ylabel('W')
    #plt.xlabel('Steps')
    train_data, train_targets, test_data, test_targets = load_data()
    train_data = np.concatenate((train_data, np.ones(shape=(np.size(train_data, 0), 1))), axis=1)
    test_data = np.concatenate((test_data, np.ones(shape=(np.size(test_data, 0), 1))), axis=1)


    #train SVM
    svm_model = optimize_svm(train_data, train_targets, 1.0, 0.1, 100, 500)

    #get loss
    train_loss = svm_model.hinge_loss(train_data,train_targets)
    print('The training loss: ', train_loss.mean())
    test_loss = svm_model.hinge_loss(test_data, test_targets)
    print('The test loss: ', test_loss.mean())

    #get accuracy
    train_accuracy = svm_model.classify(train_data)
    print('Training accuracy = {}'.format((train_accuracy == train_targets).mean()))
    test_accuracy = svm_model.classify(test_data)
    print('Test accuracy = {}'.format((test_accuracy == test_targets).mean()))

    #Plot W
    w = svm_model.w[0:784]
    img_w = w.reshape((28, 28))
    plt.imshow(img_w,cmap='gray')
