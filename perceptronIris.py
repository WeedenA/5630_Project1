from sklearn.datasets import load_svmlight_file
import numpy as np


class Perceptron:

    # Initialize perceptron
    # Learning rate scalar, iteration count, and weights/bias
    def __init__(self, scalar=0.01, iter=1000):
        self.scalar = scalar
        self.iter = iter
        #self.stepFunc = self.unitStepFunc()
        self.weight = None
        self.bias = None

    # Fits hyperplane to provided data to test against y, a vector of 1's and -1's
    def fit(self, X, y):
        samples, features = X.shape

        self.weight = np.zeros(features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.iter):
            for idx, x in enumerate(X):
                output = np.dot(x, self.weight) + self.bias
                predictedY = self.unitStepFunc(output)
                update = self.scalar * (y_[idx] - predictedY)
                self.weight += update * x
                self.bias += update

    # Predicts class of y based on current weights and biases
    def predict(self,X):
        linearOutput = np.dot(X, self.weight) + self.bias
        predictedY = self.unitStepFunc(linearOutput)
        return predictedY

    def unitStepFunc(self,x):
        return np.where(x>1,x,2)

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
np.random.shuffle(xdf)
np.random.shuffle(ydf)

testSize = 0.15
xTrain = xdf[:-int(testSize*len(xdf))]
yTrain = ydf[:-int(testSize*len(ydf))]
xTest = xdf[-int(testSize*len(xdf)):]
yTest = ydf[-int(testSize*len(ydf)):]
perceptron = Perceptron()
perceptron.fit(np.array(xTrain), yTrain)
prediction = np.array(perceptron.predict(np.array(xTest)))
actual = np.array(yTest)

total = 0
correct = 0
for i in range(len(prediction)):
    total +=1
    if int(prediction[i]) == int(actual[i]):
        correct +=1

print(correct/total)