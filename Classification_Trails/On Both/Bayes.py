# importing packages
import pandas as pd
import numpy as np
# preprocessing the data
data1 = pd.read_csv('spam.csv', encoding = 'latin-1')
data1 = data1.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)

data2 = pd.ExcelFile('revisedindiandataset.xls')
data2 = data2.parse(0)
data2 = data2.drop(['code'], axis = 1)


data = pd.concat([data1, data2])
# encoding Data
data['label'] = pd.factorize(data['label'])[0]


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
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train.toarray(), y_train)
y_pred = classifier.predict(X_test.toarray())

#Confusion Matrix                                               [724 470]
#                                                               [10  189]   65.54
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)