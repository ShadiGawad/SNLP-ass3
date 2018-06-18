import csv
from scipy.sparse import csr_matrix
import numpy
from collections import Counter
from sklearn import linear_model


#Exercise 2
#method to read the lables of the csv file
def get_lables(file, lables=[]):
    with open(file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter= ",")
        next(readCSV, None)
        for row in readCSV:
            lables.append(row[0])
        return lables

#returns all the bigrams of a string
def get_bigrams(s):
    return [s[i:i+2] for i in range(len(s)-1)]

#set up counts list, languages for all lables and a set of all bigrams
counts = []
languages = []
bigrams=set()

#read the csv file
with open("tweet-corpus.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter = ",")
    next(readCSV, None)                             #skip header
    languages = get_lables("tweet-corpus.csv")
    for row in readCSV:
        c = Counter(get_bigrams(row[2]))
        counts.append(c)
        bigrams |= set(c.keys())

bigrams = sorted(bigrams)

matrixRows = len(counts)
matrixColumns = len(bigrams)
matrix = numpy.zeros((matrixRows, matrixColumns))

print("Ex2 matrix")
print(matrix)

for i,t in enumerate(counts):
    for k,v in t.most_common():
        j = bigrams.index(k)
        matrix[i,j] = v

#Exercise 3
def logisticRegressionTrainer(matrix, lables, c):
    model = linear_model.LogisticRegression(C=c)
    model.fit(matrix, lables)
    return model


score= logisticRegressionTrainer(matrix,languages,1.0).score(matrix,languages)
print("Ex3 score:",score)

#Exercise 4

predictions = logisticRegressionTrainer(matrix,languages,1.0).predict(matrix)

def prf(gold, pred, stringCheck):
    TP = 0
    FP = 0
    FN = 0
    for i,j in zip(gold,pred):
        if i == stringCheck and j == stringCheck:
            TP += 1
        elif j == stringCheck and i != stringCheck:
            FP += 1
        elif i == stringCheck and j != stringCheck:
            FP += 1

        if TP == 0:
            TP = 1
    precision = (TP / (TP + FP))
    recall = (TP / (TP + FN))
    f1 = (2* precision * recall)/(precision + recall)

    return precision, recall, f1

def macro(prediction, langList):
    langListUnique = list(set(langList))
    c = len(langListUnique)

    sumPrecision = 0
    sumRecall = 0

    for langString in langListUnique:
        p,r,f = prf(languages,prediction,langString)
        sumPrecision += p
        sumRecall += r

    macro_precision = sumPrecision / c
    macro_recall = sumRecall / c
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

    return macro_precision, macro_recall, macro_f1

print("Ex4: Macro average precision, recall and f1:", macro(predictions, languages))

#Exercise 5

def kfold(data,lables,k,c):
    partition = int(len(data) / k)
    start = 0

    randomizer = numpy.random.permutation(len(data))
    random_data = data[randomizer]
    random_lables = numpy.array(lables)
    random_lables = random_lables[randomizer]


    macroPrecision = []
    macroRecall = []
    macroF1 = []

    for i in range(k):
        if(len(random_data) - start) <= partition:
            testData = random_data[start:]
            testLables = random_lables[start:]
            trainingData = numpy.delete(random_data, numpy.s_[start:],axis = 0)
            trainingLables = numpy.delete(random_lables, numpy.s_[start:])
        else:
            testData = random_data[start:(start+partition)]
            testLables = random_lables[start:(start+partition)]
            trainingData = numpy.delete(random_data,numpy.s_[start:(start+partition)],axis=0)
            trainingLables = numpy.delete(random_lables, numpy.s_[start:(start+partition)])
            start = partition

        kFoldModel = logisticRegressionTrainer(trainingData, trainingLables,c)
        kFoldPrediction = kFoldModel.predict(testData)
        mP, mR, mF = macro(kFoldPrediction, testLables)

    macroPrecision.append(mP)
    macroRecall.append(mR)
    macroF1.append(mF)
    mpMean = numpy.mean(macroPrecision)
    mrMean = numpy.mean(macroRecall)
    mf1Mean = numpy.mean(macroF1)

    return mpMean,mrMean,mf1Mean

print("Ex5: Mean of Macro Precision, Recall and F1: ", kfold(matrix,languages, 5,1.0))

def maxF (data):
    result = [0,0,0,0]

    for item in data:
        if result[3] <= item[3]:
            result[0] = item[0]
            result[1] = item[1]
            result[2] = item[2]
            result[3] = item[3]

    return result

#Exercise 6.

hyper = 0.0
result = []
result2 = []
data = []
no_k_fold =[]

while hyper < 2.0:
    hyper += 0.5

    hPrecision, hRecall, hF1 = kfold(matrix, languages,5,hyper)
    result.append([hyper,hPrecision,hRecall,hF1])

    improvedModel = linear_model.LogisticRegression( C=hyper)
    fitImprovedModel= improvedModel.fit(matrix,languages)
    prediction = fitImprovedModel.predict(matrix)
    precisionImproved, recallImproved, fImproved = macro(prediction,languages)
    result2.append([hyper, precisionImproved, recallImproved,fImproved])

data = maxF(result)
no_k_fold = maxF(result2)

print("Ex6 with kfold: Hyper_Parameter, Macro precision, Macro Recall, Macro F1:", data)
print("Ex6 without kfold: Hyper_Parameter, Macro precision, Macro Recall, Macro F1:", no_k_fold)

