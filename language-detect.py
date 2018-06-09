import csv
from scipy.sparse import csr_matrix
import numpy
from collections import Counter
from sklearn import linear_model

def get_bigrams(s):
    return [s[i:i+2] for i in range(len(s)-1)]

counts = []
languages = []
bigrams=set()

with open("tweet-corpus.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter = ",")
    next(readCSV, None)

    for row in readCSV:
        languages.append(row[0])
        c = Counter(get_bigrams(row[2]))
        counts.append(c)
        bigrams |= set(c.keys())

bigrams = sorted(bigrams)

matrixRows = len(counts)
matrixColumns = len(bigrams)
print(matrixColumns)
print(matrixRows)
matrix = numpy.zeros((matrixRows, matrixColumns))

for i,t in enumerate(counts):
    for k,v in t.most_common():
        j = bigrams.index(k)
        matrix[i,j] = v

for i,t in zip(matrix,languages):
    print (t, i)

y_train = matrix
x_train = numpy.array(languages).reshape(-1,1)

print(x_train)
model = linear_model.LinearRegression()
model.fit(x_train,y_train)

