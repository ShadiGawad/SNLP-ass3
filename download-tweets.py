from sys import argv
import tweepy
from tweepy import OAuthHandler
from tweepy import API
import csv


cKey= "8uDqFccWC99TpW6baa7N2SPDo"
cSecret= "bpI8Gm64AeaE9PLUj545nzHto0gaeEPkdZhv2PhedVo1eqZEsE"
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