import csv
from scipy.sparse import csr_matrix
import numpy
from collections import Counter
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB


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



#read the csv file
def get_Matrix_lables(filename):
    # set up counts list, languages for all lables and a set of all bigrams
    counts = []
    languages = []
    bigrams = set()
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter = ",")
        next(readCSV, None)                             #skip header
        languages = get_lables("tweet-corpus.csv")
        for row in readCSV:
            c = Counter(get_bigrams(row[2]))
            counts.append(c)
            bigrams |= set(c.keys())

    bigrams = sorted(bigrams)       #get a list of unique bigrams

    matrixRows = len(counts)
    matrixColumns = len(bigrams)
    matrix = numpy.zeros((matrixRows, matrixColumns))

    for i,t in enumerate(counts):
        for k,v in t.most_common():
            j = bigrams.index(k)
            matrix[i,j] = v

    print("Ex2 matrix")
    print(matrix)
    return matrix, languages

matrix, languages = get_Matrix_lables("tweet-corpus.csv")

#Exercise 3
#method to train a logistic regression model with our matrix and our language lables of the csv file
def logisticRegressionTrainer(matrix, lables, c):
    model = linear_model.LogisticRegression(C=c)
    model.fit(matrix, lables)
    return model

#get the score of the model
score= logisticRegressionTrainer(matrix,languages,1.0).score(matrix,languages)
print("Ex3 score:",score)

#Exercise 4

predictions = logisticRegressionTrainer(matrix,languages,1.0).predict(matrix)
#method which returns precision recall and f1 score
#needs two lists, gold and prediction lables and a lable to check
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
    #calculation
    precision = (TP / (TP + FP))
    recall = (TP / (TP + FN))
    f1 = (2* precision * recall)/(precision + recall)

    return precision, recall, f1

def macro(prediction, langList):
    langListUnique = list(set(langList))
    #get the length of all unique languages
    c = len(langListUnique)

    sumPrecision = 0
    sumRecall = 0

    #calculate the sum of all precisions and recalls
    for langString in langListUnique:
        p,r,f = prf(languages,prediction,langString)
        sumPrecision += p
        sumRecall += r

    #calculate macro precision recall and f1
    macro_precision = sumPrecision / c
    macro_recall = sumRecall / c
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

    return macro_precision, macro_recall, macro_f1

print("Ex4: Macro average precision, recall and f1:", macro(predictions, languages))

#Exercise 5

def kfold(data,lables,k,c):
    #create a integer to track the parts of the data
    partition = int(len(data) / k)
    start = 0

    #shuffle data and lables
    randomizer = numpy.random.permutation(len(data))
    random_data = data[randomizer]
    random_lables = numpy.array(lables)
    random_lables = random_lables[randomizer]


    macroPrecision = []
    macroRecall = []
    macroF1 = []

    for i in range(k):
        #if there is something missing after the training data we take out
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
        #train new model with our training data and lables
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
#method to calculate the highest f1 score
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

#try different hyper parameters
while hyper < 2.0:
    hyper += 0.5

    #results with kfold
    hPrecision, hRecall, hF1 = kfold(matrix, languages,5,hyper)
    result.append([hyper,hPrecision,hRecall,hF1])

    #new model with updated hyper parameter 'C'
    improvedModel = linear_model.LogisticRegression( C=hyper)
    fitImprovedModel= improvedModel.fit(matrix,languages)
    prediction = fitImprovedModel.predict(matrix)
    #results without kfold
    precisionImproved, recallImproved, fImproved = macro(prediction,languages)
    result2.append([hyper, precisionImproved, recallImproved,fImproved])

#calculate the highest f1 score for both results
data = maxF(result)
no_k_fold = maxF(result2)

print("Ex6 with kfold: Hyper_Parameter, Macro precision, Macro Recall, Macro F1:", data)
print("Ex6 without kfold: Hyper_Parameter, Macro precision, Macro Recall, Macro F1:", no_k_fold)

def alternative_prediction(filename):
    #get the matrix and the lables of the training set
    trainingMatrix, trainingLables = get_Matrix_lables("testset-sample+gold.csv")

    #get matrix and lables from the test set
    testMatrix, testLables = get_Matrix_lables(filename)

    tweets = []

    #get tweet texts from the file
    with open(filename, "r") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            tweet = row[2]
            tweets.append(tweet)


    #initialize multinomial naive bayes model and get predictions
    multiModel = MultinomialNB
    multiFitting = multiModel.fit(trainingMatrix,trainingLables)
    multiPrediction = multiFitting.predict(testMatrix)

    #print predictions and texts into the csv file
    csvfile = open('predictions.csv', 'w')
    header = ['Language', 'Id', 'Text']
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writerow({'Language': multiPrediction, 'ID': "_", 'Text': tweets})


alternative_prediction("testset-sample.csv")