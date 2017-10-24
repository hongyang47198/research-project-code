file_path = "all_time.txt"

ada_id = []
ada_time = []
mel_id = []
mel_time = []
syd_id = []
syd_time = []

with open("target_ada_coor.txt",'r') as f:
    for line in f:
        ada_id.append(line.split()[0])

with open("target_mel_coor.txt",'r') as f:
    for line in f:
        mel_id.append(line.split()[0])

with open("target_syd_coor.txt",'r') as f:
    for line in f:
        syd_id.append(line.split()[0])


with open("ada_time.txt",'wb') as f:
    for line in ada_time:
        f.write(line)


with open("mel_time.txt",'wb') as f:
    for line in mel_time:
        f.write(line)

with open("syd_time.txt",'wb') as f:
    for line in syd_time:
        f.write(line)
        f.write('\n')

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

flu_keywords = set([u'flu',u'Flu',u'ill',u'sick',u'bad',u'cold',u'fever',u'unwell',u'queasy',u'feverish',u'disease',u'infected',u'suffering',u'ache','disorder','syndrome','dose','condition','bug','indisposition','malady','ailment','drug',u'infection',u'immune',u'incubation',u'precaution',u'influenza',u'vaccine',u'virus',u'virulent',u'dose',u'indisposition',u'infirmity',u'disease',u'ache',u'illness',u'chill',u'cough',u'runny',u'sneeze',u'sniff',u'sniffle',u'snuffle',u'tickle',u'frog',u'catarrh',u'chesty',u'vomit'])
chicken_keywords = set([u'pox',u'chicken',u'sick',u'fever',u'unwell',u'queasy',u'feverish',u'disease',u'infected',u'suffering',u'ache','disorder','syndrome','dose','condition','indisposition','malady','ailment','drug',u'infection',u'immune',u'incubation',u'precaution',u'influenza',u'vaccine',u'virus',u'virulent',u'dose',u'indisposition',u'infirmity',u'disease',u'ache',u'illness',u'chill',u'cough',u'sniff',u'sniffle',u'snuffle',u'tickle',u'frog',u'catarrh',u'chesty',u'vomit'])
miseal_keywords = set([u'ill',u'sick',u'bad',u'fever',u'unwell',u'queasy',u'feverish',u'disease',u'infected',u'suffering',u'ache','disorder','syndrome','dose','condition','bug','indisposition','malady','ailment','drug','measles',u'infection',u'immune',u'incubation',u'precaution',u'influenza',u'vaccine',u'virus',u'virulent',u'dose',u'indisposition',u'infirmity',u'disease',u'ache',u'illness',u'chill',u'sniff',u'sniffle',u'snuffle',u'tickle',u'frog',u'catarrh',u'chesty',u'vomit'])


lemmatizer = WordNetLemmatizer()
word_list = set(w.lower() for w in words.words())
stop_words = set(stopwords.words('english'))
punc = string.punctuation

coor_id = []
all_coor = []

text_id = []
all_text =[]

n = 0
with open("result_text.txt",'rb') as f:
    for line in f:
        s = line.split()
        text_id.append(s[0])
        all_text.append(s[1:])
        n += 1


flu_target = []
flu_target_text = []
miseal_target = []
miseal_target_text = []
pox_target = []
pox_target_text = []
count = 0
for n in range(len(all_text)):
    for word in all_text[n]:
        word = re.sub("[^A-za-z]","",word)
        if word and word.lower() in flu_keywords:
            flu_target.append(text_id[n])
            flu_target_text.append(all_text[n])
            break
for n in range(len(all_text)):
    for word in all_text[n]:
        word = re.sub("[^A-za-z]","",word)
        if word and word.lower() in chicken_keywords:
            pox_target.append(text_id[n])
            pox_target_text.append(all_text[n])
            break
for n in range(len(all_text)):
    for word in all_text[n]:
        word = re.sub("[^A-za-z]","",word)
        if word and word.lower() in miseal_keywords:
            miseal_target.append(text_id[n])
            miseal_target_text.append(all_text[n])
            break
    count += 1
    if count%1000000 == 0:
        print count

flu_coor = []
mi_coor = []
pox_coor = []
pox_target_set = set(pox_target)
flu_target_set = set(flu_target)
miseal_target_set = set(miseal_target)
with open("result_coordinate.txt",'rb') as f:
    for line in f:
        s = line.split()[0]
        if s in flu_target_set:
            flu_coor.append(line)
        if s in miseal_target_set:
            mi_coor.append(line)
        if s in pox_target_set:
            pox_coor.append(line)
mel_flu = []
mel_flu_t= []
mel_mi = []
mel_mi_t =[]
mel_pox = []
mel_pox_t = []
ada_flu = []
ada_flu_t = []
ada_mi = []
ada_mi_t = []
ada_pox = []
ada_pox_t = []
syd_flu = []
syd_flu_t = []
syd_mi = []
syd_mi_t = []
syd_pox = []
syd_pox_t =[]

for coor in flu_coor:
    s = coor.split()
    t = s[0]
    if len(s[1]) > 6:
        f_coordinate = int(s[1][2:5])
        s_coordinate = int(s[2][2:4])
#         print f_coordinate,s_coordinate
        if f_coordinate >= 138 and f_coordinate<= 139 and s_coordinate <= 35 and s_coordinate >= 34:
            ada_flu.append(coor)
            ada_flu_t.append(t)
        if f_coordinate >=143 and f_coordinate <= 146 and s_coordinate <=38 and s_coordinate >= 37:
            mel_flu.append(coor)
            mel_flu_t.append(t)
        if f_coordinate >=150 and f_coordinate <= 152 and s_coordinate <=34 and s_coordinate >= 33:
            syd_flu.append(coor)
            syd_flu_t.append(t)

# print ada_flu,mel_flu,syd_flu
for coor in mi_coor:
    s = coor.split()
    t = s[0]
    if len(s[1]) > 6:
        f_coordinate = int(s[1][2:5])
        s_coordinate = int(s[2][2:4])
        if f_coordinate >= 138 and f_coordinate<= 139 and s_coordinate <= 35 and s_coordinate >= 34:
            ada_mi.append(coor)
            ada_mi_t.append(t)
        if f_coordinate >=143 and f_coordinate <= 146 and s_coordinate <=38 and s_coordinate >= 37:
            mel_mi.append(coor)
            mel_mi_t.append(t)
        if f_coordinate >=150 and f_coordinate <= 152 and s_coordinate <=34 and s_coordinate >= 33:
            syd_mi.append(coor)
            syd_mi_t.append(t)

for coor in pox_coor:
    s = coor.split()
    t = s[0]
    if len(s[1]) > 6:
        f_coordinate = int(s[1][2:5])
        s_coordinate = int(s[2][2:4])
        if f_coordinate >= 138 and f_coordinate<= 139 and s_coordinate <= 35 and s_coordinate >= 34:
            ada_pox.append(coor)
            ada_pox_t.append(t)
        if f_coordinate >=143 and f_coordinate <= 146 and s_coordinate <=38 and s_coordinate >= 37:
            mel_pox.append(coor)
            mel_pox_t.append(t)
        if f_coordinate >=150 and f_coordinate <= 152 and s_coordinate <=34 and s_coordinate >= 33:
            syd_pox.append(coor)
            syd_pox_t.append(t)

mel_flu_time= []
mel_mi_time =[]
mel_pox_time = []
ada_flu_time = []
ada_mi_time = []
ada_pox_time = []
syd_flu_time = []
syd_mi_time = []
syd_pox_time =[]

mel_flu_t= set(mel_flu_t)
mel_mi_t =set(mel_mi_t)
mel_pox_t = set(mel_pox_t)
ada_flu_t = set(ada_flu_t)
ada_mi_t = set(ada_mi_t)
ada_pox_t = set(ada_pox_t)
syd_flu_t = set(syd_flu_t)
syd_mi_t = set(syd_mi_t)
syd_pox_t =set(syd_pox_t)
with open('all_time.txt','rb') as f:
    n = 0
    for line in f:
        s = line.split()[0]
        if s in mel_flu_t:
            mel_flu_time.append(line)
        if s in mel_mi_t:
            mel_mi_time.append(line)
        if s in mel_pox_t:
            mel_pox_time.append(line)
        if s in ada_flu_t:
            ada_flu_time.append(line)
        if s in ada_mi_t:
            ada_mi_time.append(line)
        if s in ada_pox_t:
            ada_pox_time.append(line)
        if s in syd_flu_t:
            syd_flu_time.append(line)
        if s in syd_mi_t:
            syd_mi_time.append(line)
        if s in syd_pox_t:
            syd_pox_time.append(line)
        if n%5000 == 0:
            print n
        n += 1

with open("mel/flu_time.txt",'w') as f:
    for line in mel_flu_time:
        f.write(line)

with open("mel/mi_time.txt",'w') as f:
    for line in mel_mi_time:
        f.write(line)
with open("mel/pox_time.txt",'w') as f:
    for line in mel_pox_time:
        f.write(line)

with open("ada/flu_time.txt",'wb') as f:
    for line in ada_flu_time:
        f.write(line)
with open("ada/mi_time.txt",'wb') as f:
    for line in ada_mi_time:
        f.write(line)
with open("ada/pox_time.txt",'wb') as f:
    for line in ada_pox_time:
        f.write(line)

with open("syd/flu_time.txt",'wb') as f:
    for line in syd_flu_time:
        f.write(line)
with open("syd/mi_time.txt",'wb') as f:
    for line in syd_mi_time:
        f.write(line)
with open("syd/pox_time.txt",'wb') as f:
    for line in syd_pox_time:
        f.write(line)

with open("mel/flu.txt",'w') as f:
    for line in mel_flu:
        f.write(line)

with open("mel/mi.txt",'w') as f:
    for line in mel_mi:
        f.write(line)
with open("mel/pox.txt",'w') as f:
    for line in mel_pox:
        f.write(line)

with open("ada/flu.txt",'wb') as f:
    for line in ada_flu:
        f.write(line)
with open("ada/mi.txt",'wb') as f:
    for line in ada_mi:
        f.write(line)
with open("ada/pox.txt",'wb') as f:
    for line in ada_pox:
        f.write(line)

with open("syd/flu.txt",'wb') as f:
    for line in syd_flu:
        f.write(line)
with open("syd/mi.txt",'wb') as f:
    for line in syd_mi:
        f.write(line)
with open("syd/pox.txt",'wb') as f:
    for line in syd_pox:
        f.write(line)

coordinate_file_name = "data/result_coordinate.txt"
test = "data/test_coordinates.txt"


# Melbourne1 -38 144   -38 146   -37 144   -37 146
# Adelaide   -35 138   -35 139   -34 138   -34 139
# Sydney     -34 151   -34 152   -33 151   -33 152

mel = []
mel_coor = []
ada = []
ada_coor = []
syd = []
syd_coor = []

with open(coordinate_file_name,'r') as cf:
    count_ad = 0
    count_mel = 0
    count_sy = 0
    count =0
    for line in cf:
        count += 1
        s = line.split()
        if len(s[1]) > 6:
            f_coordinate = int(s[1][2:5])
            s_coordinate = int(s[2][2:4])
            if f_coordinate >= 138 and f_coordinate<= 139 and s_coordinate <= 35 and s_coordinate >= 34:
                ada_coor.append(s[0])
                ada.append(line)
                count_ad +=1
            if f_coordinate >=143 and f_coordinate <= 146 and s_coordinate <=38 and s_coordinate >= 37:
                mel_coor.append(s[0])
                mel.append(line)
                count_mel +=1
            if f_coordinate >=150 and f_coordinate <= 152 and s_coordinate <=34 and s_coordinate >= 33:
                syd_coor.append(s[0])
                syd.append(line)
                count_sy +=1
    cf.close()
    print count_ad
    print count_mel
    print count_sy
    print count


