from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np



def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        # TODO: Plot feature i against y
        plt.scatter(X[:, i], y)
        plt.xlabel(features[i])
        plt.ylabel('MEDV')
        plt.tight_layout()
    plt.show()


def fit_regression(X, Y):
    # TODO: implement linear regression

    # Remember to use np.linalg.solve instead of inverting!
    data = np.matrix(X)
    target = np.matrix(Y).transpose()
    data_t = data.transpose()
    #Using normal equation
    result = np.dot(np.dot(np.linalg.solve(np.dot(data_t,data),np.identity(data.shape[1])),data_t),target)

    return result



    raise NotImplementedError()



def to_predect(w, data):
    result = np.dot(np.matrix(w).transpose(),np.matrix(data).transpose())[0,0]
    return result

def compute_mse(test_data,test_target,w):
    result = 0;
    for s in range(len(test_target)):
        result = result + (to_predect(w, test_data[s]) - test_target[s])**2

    return result/len(test_target)

def compute_mae(test_data,test_target,w):
    result = 0;
    for s in range(len(test_target)):
        result = result + abs(to_predect(w, test_data[s]) - test_target[s])

    return result/len(test_target)



def main():
    # Load the data
    X, y, features = load_data()
    print("Number of data points: {}\nDimensions: {}\nTarget: Median Value (attribute 14)".format(str(X.shape[0]),str(X.shape[1])))

    # Visualize the features
    visualize(X, y, features)

    # TODO: Split data into train and test
    train_index = np.random.choice(X.data.shape[0],int(X.data.shape[0]*0.8), replace=False)
    train_data = []
    train_target = []
    test_data = []
    test_target = []
    new_X = np.insert(X, 0, 1, axis=1)
    for s in range(X.shape[0]):
        if s in train_index:
            train_data.append(new_X[s])
            train_target.append(y[s])
        else:
            test_data.append(new_X[s])
            test_target.append(y[s])
    train_target = np.array(train_target)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    test_target =np.array(test_target)
    # Fit regression model
    w = fit_regression(train_data, train_target)
    print(w)

    # Compute fitted values, MSE, etc.
    MSE = compute_mse(test_data,test_target,w)
    RMSE = np.sqrt(MSE)
    MAE = compute_mae(test_data,test_target,w)
    print("MSE: {}".format(MSE))
    print("RMSE: {}".format(RMSE))
    print("MAE: {}".format(MAE))
if __name__ == "__main__":
    main()
