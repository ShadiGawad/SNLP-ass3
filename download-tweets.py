from sys import argv
import tweepy
from tweepy import OAuthHandler
from tweepy import API
import csv


cKey= "CG6yV4x3jeTff2I49DjjJHrSQ"
cSecret= "Xd94kql1mlIyxT1kqeUGTXP7PbylBZMunsZIjeVU6xGSpPkVyG"
aToken= "950698276143026176-tsM24sGDO8S94tME52g2ySEa47Ne8PR"
aSecret= "xpRvwgpVNJ16yiyjaBsGbKpzndCz59nzjCjUaW1ovqfW2"
pin = "4135991"

auth = OAuthHandler(cKey, cSecret)
auth.set_access_token(aToken, aSecret)
api = API(auth)
ids = []
column=['language','ID',]

def main():
    files = argv[1:]
    csvfile = open('tweet-corpus.csv','w')

    for file in files:
        data = open(file,'r')

        for line in file:
            ids.append(line.strip())

            for id in ids:
                tweet = api.get_status(id)





if __name__ == '__main__':
    main()