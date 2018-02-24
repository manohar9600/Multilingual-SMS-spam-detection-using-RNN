import pandas as pd
import numpy as np
import sys

data = pd.read_csv('Resources/spam.csv', encoding='latin-1')
data.head()

list = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
data = data.drop(list, axis=1)

from gensim.models import KeyedVectors

# loading the downloaded model
model = KeyedVectors.load_word2vec_format(
    'Resources/GoogleNews-vectors-negative300.bin', binary=True)


def breakdot(word):
    tem = []
    fa = 0
    for i in range(len(word)):
        if word[i] == '.':
            fa = fa + 1
        if fa == 2:
            break
    if fa < 2:
        tem = word.split('.')
        return tem
    else:
        tem.append(word)
        return tem


data2vec = []
for row in data.itertuples():
    temp = []
    words = row[2].lower()
    words = words.split()
    for word in words:
        te = breakdot(word)
        for q in te:
            try:
                mod = model[q]
                temp.append(mod)
            except Exception:
                pass
    data2vec.append(temp)
