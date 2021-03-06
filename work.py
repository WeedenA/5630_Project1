from sklearn.datasets import load_svmlight_file
import numpy as np
from collections import Counter

#
# Perceptron implementation
#
class CustomPerceptron(object):

    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate

    '''
    Stochastic Gradient Descent 

    1. Weights are updated based on each training examples.
    2. Learning of weights can continue for multiple iterations
    3. Learning rate needs to be defined
    '''

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        for _ in range(self.n_iterations):
            for xi, expected_value in zip(X, y):
                predicted_value = self.predict(xi)
                self.coef_[1:] = self.coef_[1:] + self.learning_rate * (expected_value - predicted_value) * xi
                self.coef_[0] = self.coef_[0] + self.learning_rate * (expected_value - predicted_value) * 1

    '''
    Net Input is sum of weighted input signals
    '''

    def net_input(self, X):
        weighted_sum = np.dot(X, self.coef_[1:]) + self.coef_[0]
        return weighted_sum

    '''
    Activation function is fed the net input and the unit step function
    is executed to determine the output.
    '''

    def activation_function(self, X):
        weighted_sum = self.net_input(X)
        return np.where(weighted_sum >= 0.0, 1, 0)

    '''
    Prediction is made on the basis of output of activation function
    '''

    def predict(self, X):
        return self.activation_function(X)

    '''
    Model score is calculated based on comparison of 
    expected value and predicted value
    '''

    def score(self, X, y):
        misclassified_data_count = 0
        for xi, target in zip(X, y):
            output = self.predict(xi)
            if (target != output):
                misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count) / total_data_count
        return self.score_

xData, yData = load_svmlight_file('iris.scale')

xdf = []
for i in range(xData.shape[0]):
    tempList = []
    for j in range(4):
        tempList.append(xData[i,j])
    xdf.append(tempList)
ydf = []
for i in range(yData.shape[0]):
    ydf.append(yData[i])

testSize = 0.3
xTrain = xdf[:-int(testSize*len(xdf))]
yTrain = ydf[:-int(testSize*len(ydf))]
xTest = xdf[-int(testSize*len(xdf)):]
yTest = ydf[-int(testSize*len(ydf)):]

perc = CustomPerceptron()
perc.fit(np.array(xTrain),np.array(yTrain))
print(perc.score(np.array(xTest),np.array(yTest)))



