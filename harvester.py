import couchdb
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import re

print "Begin to run the script"

ckey = "5tEFHFFDWnDhjLnf6UMmBZQkQ"
csecret = "Z1DtSrcF86v8CQjmixZUyXixSjNOY3OCBqshVPczA2nFPs1OLU"
atoken = "2593822183-Xdj8bzb8gntN0hRorsUFQPmTbMednt947WU4sgR"
asecret = "HjLctRCgQr3qrALcqicIer15Lf0PV0mXvwCTcLUs5XifH"


keyWords = set([u'flu',u'Flu',u'ill',u'sick',u'bad',u'chickenpox',u'cold',u'fever',u'unwell',u'queasy',u'feverish',u'disease',u'infected',u'suffering',u'ache','disorder','syndrome','dose','condition','bug','indisposition','malady','ailment','drug','measles'])

# test = set([u'is',u'are'])
regex = r'\w+'

class listener(StreamListener):
  def on_data(self, data):
    dictTweet = json.loads(data)
    # words = re.findall(regex,dictTweet['text'])
    # for word in keyWords:
    #   if word in words:
    # if u's' in dictTweet['text']:
        # print "SAVED" + str(doc) +"=>" + str(data)
    try:
      # if dictTweet["coordinates"] != None:
      # dictTweet["_id"] = str(dictTweet['id'])
      doc = db.save(dictTweet)
      print "SAVED" + str(data)
    except:
      print "Already exists"
      pass
      # break
    return True

  def on_error(self, status):
    print status

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
while True:
  try:
    twitterStream = Stream(auth, listener())
    server = couchdb.Server('http://localhost:5984/')
    try:
      db = server.create('new_melbourne')
    except:
      db = server['new_melbourne']
    twitterStream.filter(locations=[144.698259,-37.998670,145.281912,-37.626333])
    # twitterStream.filter(track=['chickenpox','chicken pox','flu','measles','get cold','ill','unwell','dose','aspirin','have a cold','H1N1','influenza','measle','H5N1','fever'])
  except:
    pass
  else:
    break
