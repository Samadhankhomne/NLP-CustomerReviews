 # Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\91976\Downloads\5th, 6th - NLP\5th, 6th - NLP project\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

# Cleaning the textss
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = [] 

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

"""
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
# ac = 71

# Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
# ac = 80


# SVC
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)
# ac = 80

#naive_bayes
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)
# ac = 79

from sklearn.linear_model import LogisticRegressionCV
classifier = LogisticRegressionCV()
classifier.fit(X_train, y_train)
# ac = 74

"""

# logostic regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
# ac = 81

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion_matrix:\n",cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("accuracy_score:",ac)
  
bias = classifier.score(X_train,y_train)
print("bias:",bias)

variance = classifier.score(X_test,y_test)
print('variance:',variance)

#===============================================
