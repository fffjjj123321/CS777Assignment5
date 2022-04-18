#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import re
import sys
import numpy as np
import pandas as pd
from operator import add
import time
from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
import pyspark.ml.feature as ft
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint 

sc = SparkContext.getOrCreate()

print ("Start time: ",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


training_data_path = 'SmallTrainingData.txt'
testing_data_path = 'TestingData.txt'

# training_corpus = sc.textFile(training_data_path, 1)
# testing_corpus = sc.textFile(testing_data_path, 1)
training_corpus = sc.textFile(sys.argv[1], 1)
testing_corpus = sc.textFile(sys.argv[2], 1)
training_keyAndText = training_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
testing_keyAndText = testing_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
regex = re.compile('[^a-zA-Z]')

training_keyAndListOfWords = training_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
testing_keyAndListOfWords = testing_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

training_keyAndListOfWords.cache()
numberOfLines = training_keyAndListOfWords.count()
print("Number of Lines: {}".format(numberOfLines))

allWords = training_keyAndListOfWords.flatMap(lambda x: x[1]).map(lambda x:(x,1))

allCounts = allWords.reduceByKey(lambda x,y:x+y)

topWords = allCounts.top(5000,key = lambda x:x[1])

topWordsK = sc.parallelize(range(5000))
dictionary = topWordsK.map (lambda x : (topWords[x][0], x))
dictionary.cache()


allWordsWithDocID = training_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
allDictionaryWords = allWordsWithDocID.join(dictionary)

# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
justDocAndPos = allDictionaryWords.map(lambda x:x[1])


def buildArray(listOfIndices):
    returnVal = np.zeros(5000)
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    mysum = np.sum(returnVal)
    returnVal = np.divide(returnVal, mysum)
    return returnVal

def build_zero_one_array (listOfIndices):
    returnVal = np.zeros (5000)
    for index in listOfIndices:
        if returnVal[index] == 0: returnVal[index] = 1
    return returnVal


# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()

allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0],buildArray(x[1])))
allDocsAsNumpyArrays.cache()

zeroOrOne = allDocsAsNumpyArrays.map(lambda x: [x[0],np.where(x[1] > 0,1,0)])

print("Building DF array")
dfArray = zeroOrOne.map(lambda x:x[1]).treeAggregate(np.zeros(5000),lambda x1, x2:np.add(x1,x2),lambda x1, x2:np.add(x1,x2),3)

idfArray = np.log(np.divide(np.full(5000, numberOfLines),dfArray))

numberOfDocs = training_keyAndListOfWords.map(lambda x:('Australian legal case' if x[0][:2] == 'AU' else 'Wikipedia documents',1)).aggregateByKey(0,lambda x,y:np.add(x,y),lambda x,y:np.add(x,y)).take(2)

numberOfDocs = pd.DataFrame(numberOfDocs)
numberOfDocs.columns = ['class','numbers']

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

featureRDD = allDocsAsNumpyArrays.map(lambda x: (x[0],np.multiply(x[1], idfArray),1 if x[0][:2] == 'AU' else -1))
featureRDD.cache()

AUfeatureRDD = featureRDD.filter(lambda x:x[2]==1)
WikiFeatureRDD = featureRDD.filter(lambda x:x[2]==-1)
AUnumberOfDocs = int(numberOfDocs[numberOfDocs['class']=='Australian legal case']['numbers'])
WikinumberOfDocs = int(numberOfDocs[numberOfDocs['class']=='Wikipedia documents']['numbers'])

AUfeatureRDD.cache()
WikiFeatureRDD.cache()


print("Number of AU legal case docs: {}".format(AUnumberOfDocs))
print("Number of Wiki docs: {}".format(WikinumberOfDocs))

test_allWordsWithDocID = testing_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

test_allDictionaryWords = test_allWordsWithDocID.join(dictionary)

test_justDocAndPos = test_allDictionaryWords.map(lambda x:x[1])

test_allDictionaryWordsInEachDoc = test_justDocAndPos.groupByKey()

test_allDocsAsNumpyArrays = test_allDictionaryWordsInEachDoc.map(lambda x: (x[0],buildArray(x[1])))

test_featureRDD = test_allDocsAsNumpyArrays.map(lambda x: (x[0],np.multiply(x[1], idfArray),1 if x[0][:2] == 'AU' else 0))


learningRate = 0.1
num_iteration = 20
theta = np.array([-1e-1]*5000)
reg_lambda = 1


start = time.time()
for i in range(num_iteration+1):
    sampleRDD = WikiFeatureRDD.sample(False,5*AUnumberOfDocs/WikinumberOfDocs,3368+i)
    sampleRDD = sampleRDD.union(AUfeatureRDD)
    sampleRDD.cache()
    sampleSize = sampleRDD.count()
    print("step {} : finish sampling, size = {}".format(i,sampleSize))
    r2 = np.sum(theta**2)
    gradientCost = sampleRDD.map(lambda x:(x[1],x[2],np.dot(x[1],theta))).map(lambda x:np.array([max(0,1 - x[2]*x[1]),0 if (1-x[1]*x[2] <= 0) else x[0]*x[1]])).treeAggregate(np.zeros(2),lambda x,y:np.add(x,y),lambda x,y:np.add(x,y),3)
    cost = (reg_lambda * gradientCost[0]+(r2 / 2)) / sampleSize
    gradient = (theta - reg_lambda*gradientCost[1])/sampleSize
    print("step {} : sample cost={:.8f}, gradient[0] = {:.8f}, gradient[4999] = {:.8f}".format(i,cost,gradient[0],gradient[4999]))
    theta = theta - learningRate * gradient
    
predictionRDD = test_featureRDD.map(lambda x:(x[0],(x[2],np.dot(x[1],theta)))).map(lambda x:(x[0],(x[1][0],1 if x[1][1] >=0 else 0)))

predictionResult = predictionRDD.map(lambda x:(x[1],1)).reduceByKey(lambda x,y:x+y).collect()

confusionMatrix = np.zeros([2,2],dtype = int)

for i in predictionResult:
    confusionMatrix[i[0]] = int(i[1])

print("Confusion Matrix:")
print(confusionMatrix)


prec = confusionMatrix[1,1] / (confusionMatrix[1,1]+confusionMatrix[0,1])
reca = confusionMatrix[1,1] / (confusionMatrix[1,1]+confusionMatrix[1,0])
F1_score = (2*prec*reca)/(prec+reca)


print("precision: {:.8f}, recall: {:.8f}, F1 score = {:.8f}".format(prec,reca,F1_score))

end = time.time()

print('The total time needed to train the model: %s Seconds'%(end-start))

sc.stop()
