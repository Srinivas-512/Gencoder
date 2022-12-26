from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
import pandas as pd
from itertools import islice
def window(seq, n=3):               
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

df1  = pd.read_csv("Coronaviridae.csv")
df1["label"] = 0
corona_seq = df1["sequences"]
split_corona_seq = []
for sequence in corona_seq:
    split_seq = str(["".join(ele) for ele in window(sequence)])
    split_corona_seq.append(split_seq)
df2  = pd.read_csv("Retroviridae.csv")
df2["label"] = 1
retro_seq = df2["sequences"]
split_retro_seq = []
for sequence in retro_seq:
    split_seq = str(["".join(ele) for ele in window(sequence)])
    split_retro_seq.append(split_seq)
df3  = pd.read_csv("Picornaviridae.csv")
df3["label"] = 2
picorna_seq = df3["sequences"]
split_picorna_seq = []
for sequence in picorna_seq:
    split_seq = str(["".join(ele) for ele in window(sequence)])
    split_picorna_seq.append(split_seq)
df4  = pd.read_csv("Adenoviridae.csv")
df4["label"] = 3
adeno_seq = df4["sequences"]
split_adeno_seq = []
for sequence in adeno_seq:
    split_seq = str(["".join(ele) for ele in window(sequence)])
    split_adeno_seq.append(split_seq)
df5  = pd.read_csv("Rhabdoviridae.csv")
df5["label"] = 4
rhabdo_seq = df5["sequences"]
split_rhabdo_seq = []
for sequence in rhabdo_seq:
    split_seq = str(["".join(ele) for ele in window(sequence)])
    split_rhabdo_seq.append(split_seq)
all_seq = split_corona_seq + split_retro_seq + split_picorna_seq + split_adeno_seq + split_rhabdo_seq
# 0 = corona, 1 = retro, ...
df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
vec = CountVectorizer()
train_data = vec.fit_transform(all_seq)
target_y = df["label"].to_numpy()
train , val , y_train , y_val = train_test_split(train_data,target_y,test_size=0.2,shuffle=True)
train, test, y_train, y_test = train_test_split(train,y_train,test_size=0.2,shuffle=True)
model = XGBClassifier() 
model.fit(train, y_train)
model.fit(val, y_val)
pred = model.predict(test)
print(f1_score(pred,y_test,average='micro') )