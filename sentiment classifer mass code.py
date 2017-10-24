import numpy as np
import nltk
import numpy as np
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import svm
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import string
import csv
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

train_file = "data/1600000/training.csv"
test_file = "data/1600000/test.csv"
lemmatizer = WordNetLemmatizer()
word_list = set(w.lower() for w in words.words())
stop_words = set(stopwords.words('english'))
punc = string.punctuation

# print stop_words
with open("data/1600000/punct.csv",'wb') as f:
    for word in list(punc):
        print word
        f.write(word)
        f.write('\n')
    f.close()

label = []
time = []
text = []

test_label = []
test_text = []
with open(train_file,'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for line in spamreader:
        label.append(line[0])
        text.append(line[5])
    csvfile.close()

print len(label),len(text)

with open(test_file,'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for line in spamreader:
        if line[0] != "2":
            test_label.append(line[0])
            test_text.append(line[5])
    csvfile.close()
# try:

all_text = []
n = 0
for line in text:
    sentence = []
    s = line.split()
    for word in s:
        if len(word)>2 and word not in stop_words:
            word = re.sub("[^A-Za-z]","",word)
            if word:
                try:
                    word = lemmatizer.lemmatize(word.lower())
                except:
                    word = word.lower()
                sentence.append(word)
    all_text.append(' '.join(sentence))
    n += 1

print len(text)
print len(all_text)

all_test_text = []
n = 0
for line in test_text:
    sentence = []
    s = line.split()
    for word in s:
        if len(word)>2 and word not in stop_words:
            word = re.sub("[^A-Za-z]","",word)
            if word:
                try:
                    word = lemmatizer.lemmatize(word.lower())
                except:
                    word = word.lower()
                sentence.append(word)
    all_test_text.append(' '.join(sentence))
    n += 1

print len(all_test_text)

train_input = open("data/1600000/training.pkl",'wb')
test_input = open("data/1600000/test.pkl",'wb')
pickle.dump(all_text,train_input)
pickle.dump(all_text,train_input)

vectorizer = CountVectorizer(min_df=1,binary = True)
text_model = vectorizer.fit_transform(all_text)
test_model = vectorizer.transform(all_test_text)

lg_p2 = linear_model.LogisticRegression(penalty='l2',C=1e5,solver = 'newton-cg')
lg_p1 = linear_model.LogisticRegression(penalty='l1',C=1e5,solver = 'liblinear')
lg_p2.fit(text_model,label)
lg_p1.fit(text_model,label)
predict1 = lg_p1.predict(test_model)
predict2 = lg_p2.predict(test_model)

path = open("data/1600000/lg1.pkl",'wb')
pickle.dump(lg_p1,path)
path = open("data/1600000/lg2.pkl",'wb')
pickle.dump(lg_p2,path)

predict1 = map(int,list(predict1))
predict2 = map(int,list(predict2))
test_label = map(int,list(test_label))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=3, random_state=0)
rf.fit(text_model,label)
predict = rf.predict(test_model)
predict = map(int,list(predict))
for n in range(len(test_label)):
    if test_label[n] == 4:
        test_label[n] = 1
    if predict[n] == 4:
        predict[n] =1
rf_pre = precision_score(test_label,predict)
rf_recall = recall_score(test_label,predict)
rf_f1 = f1_score(test_label,predict)
print rf_pre,rf_recall,rf_f1

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(text_model,label)
predict = nb.predict(test_model)
predict = map(int,list(predict))
for n in range(len(test_label)):
    if test_label[n] == 4:
        test_label[n] = 1
    if predict[n] == 4:
        predict[n] =1
mnb_pre = precision_score(test_label,predict)
mnb_recall = recall_score(test_label,predict)
mnb_f1 = f1_score(test_label,predict)
print mnb_pre,mnb_recall,mnb_f1
