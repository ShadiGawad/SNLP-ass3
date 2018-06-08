import csv
from scipy import sparse
import numpy
from collections import Counter

def featureExtraction(tuple):

    texts = []
    id = []
    languages = []
    with open("tweet-corpus.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter = ",")
        next(readCSV, None)

        for row in readCSV:
            languages.append(row[0])
            id.append(row[1])
            texts.append(row[2])

    print(languages)
    print(id)
    print(texts)

    bigrams=[]
    for text in texts:
        for i in range(len(text)-1):
            bigrams.append(text[i:i+2])

    print(bigrams)


    def countBigrams(numberBigrams=[],bi = bigrams):
        numberBigrams = Counter(bigrams)
        return numberBigrams

    print(countBigrams())


featureExtraction(tuple)