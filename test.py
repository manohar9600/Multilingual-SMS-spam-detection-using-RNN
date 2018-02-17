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
Y = pd.get_dummies(data['label']).values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)

batch_size = 20
model.fit(X_train, Y_train, nb_epoch = 8, batch_size = batch_size, verbose = 2)


validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):

    result = model.predict(
        X_validate[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]

    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1

    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

print("non spam _acc", pos_correct / pos_cnt * 100, "%")
print("spam_acc", neg_correct / neg_cnt * 100, "%")
