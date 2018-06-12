import csv
from scipy.sparse import csr_matrix
import numpy
from collections import Counter
from sklearn import linear_model


#Exercise 2

def get_lables(file, lables=[]):
    with open(file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter= ",")
        next(readCSV, None)
        for row in readCSV:
            lables.append(row[0])
        return lables

def get_bigrams(s):
    return [s[i:i+2] for i in range(len(s)-1)]

counts = []
languages = []
bigrams=set()

with open("tweet-corpus.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter = ",")
    next(readCSV, None)
    languages = get_lables("tweet-corpus.csv")
    for row in readCSV:
        c = Counter(get_bigrams(row[2]))
        counts.append(c)
        bigrams |= set(c.keys())

bigrams = sorted(bigrams)

matrixRows = len(counts)
matrixColumns = len(bigrams)
matrix = numpy.zeros((matrixRows, matrixColumns))

print(matrix)

for i,t in enumerate(counts):
    for k,v in t.most_common():
        j = bigrams.index(k)
        matrix[i,j] = v

#Exercise 3
model = linear_model.LogisticRegression()
model.fit(matrix,languages)
score=model.score(matrix,languages)
print(score)

#Exercise 4
def precision(list1, list2, stringCheck):
    TP = 0
    FP = 0
    for i,j in zip(list1,list2):
        if i is j is stringCheck:
            TP += 1
        elif j is stringCheck and i is not stringCheck:
            FP += 1
    return TP / TP + FP



def recall(list1, list2, stringCheck):
    TP = 0
    FN = 0
    for i,j in zip(list1,list2):
        if i is j is stringCheck:
            TP += 1
        elif i is stringCheck and j is not stringCheck:
            FN += 1
    return TP / TP + FN


def f1(list1, list2, stringCheck):
    p = precision(list1, list2, stringCheck)
    r = recall(list1, list2, stringCheck)
    return 2 * p * r / p + r

def kfolt(data):
    partition = len(data) / 5
    for i in range(4):
        l1 = data[:(partition * i)]
        l2 = data[partition * i:]
    print(len(data))
    print(len(l1))
    print(len(l2))
    return l1,l2

print(kfolt(matrix))