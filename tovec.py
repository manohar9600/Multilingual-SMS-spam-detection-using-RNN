#import pandas as pd
#import numpy as np
from collections import Counter

class convert2vec:

    def fit_texts(self, words, num_words):
        self.num_words = num_words
        te2 = []
        for i in range(len(words)):
            for j in range(len(words[i])):
                te2.append(words[i][j])
        q = Counter(te2)
        le = q.most_common(num_words - 1)
        temp = le[0:num_words - 1]
        self.words = []
        for item in temp:
            self.words.append(item[0])
        #print(self.words)

    def word2vector(self, words):
        vectors = []
        for i in range(len(words)):
            temp = []
            for word in words[i]:
                te = [0 for i in range(self.num_words)]
                te[self.num_words - 1] = 1
                for k in range(len(self.words)):
                    if word == self.words[k]:
                        te[k] = 1
                        te[self.num_words - 1] = 0
                        break
                temp.append(te)
            vectors.append(temp)
        return vectors

"""data = pd.read_csv('Resources/spam.csv', encoding = 'latin-1')

from keras.preprocessing.text import text_to_word_sequence

words = []
words.append(data['msg'][0])
words.append(data['msg'][1])
words.append(data['msg'][2])

temp = []

for row in words:
    rev = str(row)
    rev = text_to_word_sequence(rev)
    temp.append(rev)

vec = convert2vec()
vec.fit_texts(temp, 10)
na = vec.word2vector(temp)
print(na[0])

te2 = []
for i in range(len(temp)):
    for j in range(len(temp[i])):
        te2.append(temp[i][j])

q = Counter(te2)
le = list(q.keys())
print(q.most_common(3))"""
