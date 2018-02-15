import pandas as pd

data = pd.read_csv('Resources/spam.csv', encoding = 'latin-1')
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)

# word to vector
from gensim.models import KeyedVectors

#loading the downloaded model
model = KeyedVectors.load_word2vec_format(
    'Resources/GoogleNews-vectors-negative300.bin', binary=True)


vector_data = [[]]
del(vector_data[0])

for row in data.itertuples():
    rev = row[2]
    rev = rev.lower()
    rev = rev.split()
    tem = []
    for word in rev:
        tem.append(model[word])
    vector_data.append(tem)

# saving vectors
import pickle

with open('vectors.pkl', 'w') as f:
    pickle.dump(vector_data, f)

if model['jurong'] == None:
    print('hey')

#######################################

from autocorrect import spell

j = 0
words = []
for row in data.itertuples():
    print(str(j))
    j = j + 1
    tem = row[2].split()
    for word in tem:
        t = spell(word)
        if t not in words:
            words.append(t)

words.sort()

type(words)

file = open('Resources/words2.txt', 'w', encoding = 'latin-1')

for word in words:
    file.write(word)
    file.write('\n')


file = open('Resources/words2.txt', 'w', encoding='latin-1')

for word in words:
    file.write(spell(word))
    file.write('\n')

file.close()
