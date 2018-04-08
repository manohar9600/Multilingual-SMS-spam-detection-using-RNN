# importing packages
import pandas as pd
import numpy as np
# preprocessing the data
data = pd.read_csv('spam.csv', encoding = 'latin-1')
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
# encoding Data
data['label'] = pd.factorize(data['label'])[0]
# module for removing unwanted words
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

# for stemming words
from nltk.stem.porter import PorterStemmer
temp = []
for row in data.itertuples():
    # to keep a - z letters and 0 - 9
    rev = re.sub("[^0-9a-zA-Z]"," ",row[2])
    rev = rev.lower()
    rev = rev.split()
    ps = PorterStemmer()
    rev = [ps.stem(word) for word in rev if not word in set(stopwords.words("english"))]
    rev = " ".join(rev)
    temp.append(rev)
data['msg'] = temp

# splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['msg'], data['label'], test_size = 0.25)

# creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 410)
cv.fit(data['msg'])
X_train = cv.transform(X_train)
X_test = cv.transform(X_test)

# Training classifier
from sklearn.svm import SVC
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Confusion Matrix                                               [1204 2]
#                                                               [82  105]   93.96
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)