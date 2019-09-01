import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files

# nltk.download('stopwords')

reviews = load_files('txt_sentoken/')
X, y = reviews.data, reviews.target

# print('X', X)
# print('y', y)
#
# # Pickle Dataset
# with open('X_pickle', 'wb') as f:
#     pickle.dump(X, f)
#
# with open('y_pickle', 'wb') as f:
#     pickle.dump(y, f)

# # UnPickle Dataset
# with open('X_pickle', 'rb') as f:
#     X = pickle.load(f)
#
# with open('y_pickle', 'rb') as f:
#     y = pickle.load(f)

# Creating the Corpus (Data pre processing)
corpus = []
for i in range(0, len(X)):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+', ' ', review)   # remove single character
    review = re.sub(r'^[a-z]\s+', ' ', review)   # remove starting  single character
    review = re.sub(r'\s+', ' ', review)   # remove Extra Space
    corpus.append(review)
# print("Corpus: ", corpus)

# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words('english'))
# X = vectorizer.fit_transform(corpus).toarray()
# # 2000 most frequent feature used as max_feature
# # Min_df = minimum document frequen cy
#
# from sklearn.feature_extraction.text import TfidfTransformer
# transformer = TfidfTransformer()
# X = transformer.fit_transform(X).toarray()

# CountVectorizer + TfidfTransformer = TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

with open('TfidModel_pickle', 'wb') as f:
    pickle.dump(vectorizer, f)

from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size=0.2, random_state=0)

'''
The sentiment analysis task is mainly a binary classification problem to predict whether a given sentence is positive or negative.
'''

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train, sent_train)

with open('LogisticModel_pickle', 'wb') as f:
    pickle.dump(classifier, f)

sent_pred = classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
confusionM = confusion_matrix(sent_test, sent_pred)
# print("Confusion Matrix: ", '\n', confusionM)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(sent_test, sent_pred)
# print("accuracy", accuracy)

with open('LogisticModel_pickle', 'rb') as f:
    clfm = pickle.load(f)

with open('TfidModel_pickle', 'rb') as f:
    tfidm = pickle.load(f)

sample = ["You are bad Person"]
sample = tfidm.transform(sample).toarray()
print("Result: ", clfm.predict(sample))
