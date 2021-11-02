from sklearn.datasets import load_svmlight_file
import numpy as np
from collections import Counter

def kNN(trainX, trainY, test, k=5):
    distances = []
    for i in range(trainX.shape[0]):
        trainFeature = []
        for j in range(123):
            trainFeature.append(trainX[i,j])
        euclid = np.linalg.norm(np.array(trainFeature)-np.array(test))
        distances.append([euclid, trainY[i]])
    votes = [i[1] for i in sorted(distances)[:k]]
    voteResult = Counter(votes).most_common(1)[0][0]
    return voteResult

xData, yData = load_svmlight_file('a4a.t')
xData = xData[:200]
yData = yData[:200]

testSize = 0.3
xTrain = xData[:-int(testSize*xData.shape[0])]
yTrain = yData[:-int(testSize*xData.shape[0])]
xTest = xData[-int(testSize*xData.shape[0]):]
yTest = yData[-int(testSize*xData.shape[0]):]

correct = 0
total = 0

for i in range(xTest.shape[0]):
    featureList = []
    for j in range(123):
        featureList.append(xTest[i,j])
    vote = kNN(xTrain, yTrain, featureList)
    if vote == yTest[i]:
        correct +=1
    total +=1

acc = correct/total
print(acc)




