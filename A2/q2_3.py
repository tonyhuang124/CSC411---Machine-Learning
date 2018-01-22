'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    a = 2.0
    b = 2.0
    eta = np.zeros((10, 64))
    for idx, label in enumerate(train_labels):
        eta[int(label)] = eta[int(label)] + train_data[idx]
    eta = (eta + a - 1.0)/ (700.0 +a +b -2.0)
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    all_concat = []
    for i in range(10):
        img_i = class_images[i]
        img_i= img_i.reshape((8, 8))
        all_concat.append(img_i)
    all_concat = np.concatenate(all_concat, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    new_points = np.random.uniform(0.0,1.0,(10,64))

    for idx in range(len(new_points)):
        for pixel in range(0,64):
            if new_points[idx][pixel] < eta[idx][pixel]:
                new_points[idx][pixel] = 1
            else:
                new_points[idx][pixel] = 0


    generated_data = new_points
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    ans=[];
    for idx in range(len(bin_digits)):
        pob_list = [];
        for label in range(0, 10):
            pob = 1.0;
            for pixel in range(0,64):
                if bin_digits[idx][pixel] == 1:
                    pob = pob * eta[label][pixel]
                else:
                    pob = pob * (1-eta[label][pixel])
            pob_list.append(pob)
        ans.append(pob_list)
    ans = np.log(np.array(ans))
    return ans

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    P_y = 1/10
    P_x_y = np.exp(generative_likelihood(bin_digits, eta))
    P_y_x = np.zeros(P_x_y.shape)
    for i in range(len(P_x_y)):
        P_x = np.sum(P_x_y[i])*P_y
        for class_label in range(0,10):
            P_y_x[i][class_label] = (P_x_y[i][class_label]*P_y)/P_x
    ans = np.log(P_y_x)
    return ans

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute as described above and return
    ans = 0.0
    for idx,label in enumerate(labels):
        ans = ans + cond_likelihood[idx][int(label)]
    return ans/len(labels)
    # Compute as described above and return
def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    ans = []
    for idx in range(len(cond_likelihood)):
        ans.append(np.argmax(cond_likelihood[idx]))

    return np.array(ans)

def classification_accuracy(bin_digits,bin_labels,eta):
    predict_labels = classify_data(bin_digits,eta)
    correct_rate = 0;
    for idx in range(len(bin_labels)):
        if predict_labels[idx] == bin_labels[idx]:
            correct_rate = correct_rate + 1

    return correct_rate/len(bin_labels)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)
    generate_new_data(eta)
    print("average conditional log-likelihood for Training data: ",avg_conditional_likelihood(train_data,train_labels,eta))
    print("average conditional log-likelihood for Test data: ",avg_conditional_likelihood(test_data,test_labels,eta))
    print("accuracy on the train data: ", classification_accuracy(train_data,train_labels,eta))
    print("accuracy on the test data: ", classification_accuracy(test_data,test_labels,eta))
if __name__ == '__main__':
    main()
