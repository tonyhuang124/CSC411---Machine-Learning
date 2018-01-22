'''
Question 1 Skeleton Code


'''

import sklearn
import itertools
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))
    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))
    print('-----------------------------------------------------------------------------------------')
    return model


def SVM_model(train_data,train_label,test_data,test_label):
    cv_list = []
    C_list = np.random.uniform(0, 3, 10)
    C_list = np.append(C_list, 1)
    for idx in range(len(C_list)):
        pram_c = C_list[idx]
        model = svm.LinearSVC(C=pram_c)
        scores = cross_val_score(model, train_data, train_label, cv=5)
        scores = scores.mean()
        cv_list.append(scores)
        print(pram_c,'  ',scores)
    index_max = np.argmax(np.array(cv_list))
    pram_c = C_list[index_max]
    model = svm.LinearSVC(C=pram_c)
    model.fit(train_data, train_label)
    train_pred = model.predict(train_data)
    print('SVM train accuracy = {}'.format((train_pred == train_label).mean()))
    test_pred = model.predict(test_data)
    print('SVM test accuracy = {}'.format((test_pred == test_label).mean()))
    print('-----------------------------------------------------------------------------------------')
    return model


def mnb(train_data, train_label, test_data, test_label):
    # cv_list = []
    # alpha_list = np.random.uniform(0, 1, 100)
    # alpha_list = np.append(alpha_list, 0.01)
    # for idx in range(len(alpha_list)):
    #     alpha = alpha_list[idx]
    #     model = MultinomialNB(alpha)
    #     scores = cross_val_score(model, train_data, train_label, cv=10)
    #     scores = scores.mean()
    #     cv_list.append(scores)
    #
    # index_max = np.argmax(np.array(cv_list))
    # alpha = alpha_list[index_max]
    model = MultinomialNB(0.01)
    model.fit(train_data, train_label)
    train_pred = model.predict(train_data)
    print('mnb train accuracy = {}'.format((train_pred == train_label).mean()))
    test_pred = model.predict(test_data)
    y_true = test_label
    y_pred = test_pred
    con_mat = confusion_matrix(y_true, y_pred, labels=[0,1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    print('mnb test accuracy = {}'.format((test_pred == test_label).mean()))
    print('-----------------------------------------------------------------------------------------')
    return model, con_mat

def Logistic_Regression(train_data,train_label,test_data,test_label):
    cv_list = []
    C_list = np.random.uniform(0, 3, 10)
    C_list = np.append(C_list, 1)
    for idx in range(len(C_list)):
        pram_c = C_list[idx]
        model = linear_model.LogisticRegression(C=pram_c)
        scores = cross_val_score(model, train_data, train_label, cv=5)
        scores = scores.mean()
        cv_list.append(scores)
        print(pram_c,'  ',scores)
    index_max = np.argmax(np.array(cv_list))
    pram_c = C_list[index_max]
    model = linear_model.LogisticRegression(C=pram_c)
    model.fit(train_data,train_label)
    train_pred = model.predict(train_data)
    print('Logistic_Regression train accuracy = {}'.format((train_pred == train_label).mean()))
    test_pred = model.predict(test_data)
    print('Logistic_Regression test accuracy = {}'.format((test_pred == test_label).mean()))
    print('-----------------------------------------------------------------------------------------')
    return model




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == '__main__':
    train_data, test_data = load_data()
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20]
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    tf_idf_train, tf_idf_test,feature_names1 = tf_idf_features(train_data, test_data)
    #bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    #svm_model = SVM_model(tf_idf_train, train_data.target, tf_idf_test, test_data.target)
    mnb_model, con_matrix = mnb(tf_idf_train, train_data.target, tf_idf_test, test_data.target)
    plt.figure()
    plot_confusion_matrix(con_matrix, classes=labels,
                          title='Confusion matrix, without normalization')
    #reg = Logistic_Regression(tf_idf_train, train_data.target, tf_idf_test, test_data.target)