# importing packages
import pandas as pd
import numpy as np

# preprocessing the data
data = pd.read_csv('Resources/spam.csv', encoding='latin-1')
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# encoding Data
data['label'] = pd.factorize(data['label'])[0]


# module for removing unwanted words
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


nltk.download("stopwords")


# for stemming words

# add stopwords if needed

############################
train = data.drop(['label'], axis=1)
# splitting data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, data['label'],
                                                    test_size=0.25)


# Training classifier

# print(y_pred)

from sklearn.metrics import confusion_matrix
