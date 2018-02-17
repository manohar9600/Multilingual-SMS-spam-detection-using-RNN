import pandas as pd
import numpy as np

# loading Data

data = pd.read_csv('Resources/spam.csv', encoding = 'latin-1')
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)


import re
data['msg'] = data['msg'].apply(lambda x: x.lower())
data['msg'] = data['msg'].apply((lambda x: re.sub('[^a-zA-Z0-9\s]', '', x)))

# remove numbers


# converting text to vectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['msg'].values)
X = tokenizer.texts_to_sequences(data['msg'].values)
X = pad_sequences(X)

# classifier
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(
    Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary)

# splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, data['label'], test_size=0.30, random_state=42)

Y_train = pd.get_dummies(Y_train).values

batch_size = 20
model.fit(X_train, Y_train, nb_epoch = 7, batch_size = batch_size, verbose = 2)

y_out = model.predict(X_test)
Y_pred = []

for i in range(len(y_out)):
    if y_out[i][0] >= y_out[i][1]:
        Y_pred.append('ham')
    else:
        Y_pred.append('spam')

# metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
