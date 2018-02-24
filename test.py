import pandas as pd
import numpy as np
import copy

# loading Data

data = pd.read_csv('Resources/spam.csv', encoding = 'latin-1')
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)


import re
data['msg'] = data['msg'].apply(lambda x: x.lower())
data['msg'] = data['msg'].apply((lambda x: re.sub('[^a-zA-Z0-9\s]', '', x)))

# remove numbers

# splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    data['msg'], data['label'], test_size=0.30, random_state=42)

X_train2 = X_train

# converting text to vectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# training data text to vectors
max_fatures = 20000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(X_train.values)
X_train = tokenizer.texts_to_sequences(X_train.values)
X_train = pad_sequences(X_train, maxlen=100)

# getting dummies
Y_train = pd.get_dummies(Y_train).values


# classifier
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM


embed_dim = 128
lstm_out = 196


model = Sequential()
model.add(
    Embedding(max_fatures, embed_dim, input_length=X_train.shape[1]))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# training
batch_size = 20
model.fit(X_train, Y_train, nb_epoch = 2, batch_size = batch_size, verbose = 2)

# testing trained model

Y_pred = []

for i in range(len(X_test)):
    print(i)
    temp = copy.copy(X_train2)
    temp = temp.append(pd.Series(X_test.iloc[i]))
    max_fatures = 20000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(temp.values)
    tem = tokenizer.texts_to_sequences(temp.values)
    tem = pad_sequences(tem, maxlen=100)
    
    vec = tem[len(tem)-1]
    vec = vec.reshape(1,100)
    pred = model.predict(vec)
    print(pred)
    if pred[0][0] >= pred[0][1]:
        Y_pred.append('ham')
    else:
        Y_pred.append('spam')

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
