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
print("The first ten words of the vocabulary: ",dictionary.top(10, lambda x : x[1]))



allWordsWithDocID = training_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
allDictionaryWords = allWordsWithDocID.join(dictionary)

# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
justDocAndPos = allDictionaryWords.map(lambda x:x[1])


# In[9]:


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

featureRDD = allDocsAsNumpyArrays.map(lambda x: (x[0],np.multiply(x[1], idfArray),1 if x[0][:2] == 'AU' else 0))
featureRDD.cache()

AUfeatureRDD = featureRDD.filter(lambda x:x[2]==1)
WikiFeatureRDD = featureRDD.filter(lambda x:x[2]==0)
AUnumberOfDocs = int(numberOfDocs[numberOfDocs['class']=='Australian legal case']['numbers'])
WikinumberOfDocs = int(numberOfDocs[numberOfDocs['class']=='Wikipedia documents']['numbers'])

AUfeatureRDD.cache()
WikiFeatureRDD.cache()



# print("Number of AU legal case docs: {}".format(AUnumberOfDocs))
# print("Number of Wiki docs: {}".format(WikinumberOfDocs))

test_allWordsWithDocID = testing_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

test_allDictionaryWords = test_allWordsWithDocID.join(dictionary)

test_justDocAndPos = test_allDictionaryWords.map(lambda x:x[1])

test_allDictionaryWordsInEachDoc = test_justDocAndPos.groupByKey()

test_allDocsAsNumpyArrays = test_allDictionaryWordsInEachDoc.map(lambda x: (x[0],buildArray(x[1])))

test_featureRDD = test_allDocsAsNumpyArrays.map(lambda x: (x[0],np.multiply(x[1], idfArray),1 if x[0][:2] == 'AU' else 0))

featureRDD.take(1)


ml_sampleRDD = WikiFeatureRDD.sample(False,6*AUnumberOfDocs/WikinumberOfDocs,3368)
ml_sampleRDD = ml_sampleRDD.union(AUfeatureRDD)
new_featureRDD = ml_sampleRDD.map(lambda x:LabeledPoint(x[2],list(float(i) for i in x[1])))


print("Finish preparaing, start Training ",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
svd_model = SVMWithSGD.train(new_featureRDD, iterations=100)

start = time.time()
starttime = time.localtime()
print("Start Testing ",time.strftime("%Y-%m-%d %H:%M:%S", starttime))
test_new_featureRDD = test_featureRDD.map(lambda x:(x[2],list(float(i) for i in x[1])))
test_result = test_new_featureRDD.map(lambda x:(int(x[0]),svd_model.predict(x[1])))
ml_predictionResult = test_result.map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y).take(4)



ml_confusionMatrix = np.zeros([2,2])
for i in ml_predictionResult:
    ml_confusionMatrix[i[0]] = i[1]
    
print("Confusion Matrix:")
print(ml_confusionMatrix)


ml_prec = ml_confusionMatrix[1,1] / (ml_confusionMatrix[1,1]+ml_confusionMatrix[0,1])
ml_reca = ml_confusionMatrix[1,1] / (ml_confusionMatrix[1,1]+ml_confusionMatrix[1,0])
ml_F1_score = (2*ml_prec*ml_reca)/(ml_prec+ml_reca)


print("precision: {:.4f}, recall: {:.4f}, F1 score = {:.4f}".format(ml_prec,ml_reca,ml_F1_score))


end = time.time()
endtime = time.localtime()
print ("End time: ",time.strftime("%Y-%m-%d %H:%M:%S", endtime))

print('The total time needed to vectorize the data: %s Seconds'%(end-start))

sc.stop()




