# -*- coding: utf-8 -*-
"""Final Phishing Filter Training Testing.ipynb
Written and generated from https://colab.research.google.com/
Written by: Saroj Bhatta, CS '21, Alice Carson Tisdale Honors College, Claflin University
"""

#Phishng Email Filter

phishingTrainPath = "/content/PhishingTrainData_Claflin.xlsx"
phishingTestPath = "/content/spamPhishingTestData_Claflin.xlsx"

#importing important libraries
import pandas as pd
import numpy as np

##importing training data and separating columns
data = pd.read_excel(phishingTrainPath)
data = data.drop('sn', axis=1)
dataX = data.drop('target', axis=1)
subject_data = dataX.drop('message', axis=1)
message_data = dataX.drop('subject', axis=1)
dataYRaw = data.drop('subject', axis=1)
dataY = dataYRaw.drop('message', axis=1)

#download stop-words from ntlk
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

#function to clean text-data rows
def cleanText(messagesColumn):
  cleanRow = ""
  for words in messagesColumn.split():
    if len(words) > 2 and words not in stop:
      print(words)
      cleanRow = cleanRow + " " + words
  return cleanRow

#cleaning data using cleanTest function
cleanSubject = subject_data['subject'].apply(lambda row: cleanText(row))
cleanSubject = pd.DataFrame({"newSubject": cleanSubject})
cleanMessage = message_data['message'].apply(lambda row: cleanText(row))
cleanMessage = pd.DataFrame({'newMessage':cleanMessage})

#import and use vectorizer to fit and transform cleaned data
from sklearn.feature_extraction.text import CountVectorizer
vectorizerSub = CountVectorizer()
vectorizerMsg = CountVectorizer()
vectorizerSub.fit(cleanSubject['newSubject'])
vectorizerMsg.fit(cleanMessage['newMessage'])
vectorSubject = vectorizerSub.transform(cleanSubject['newSubject'])
vectorMessage = vectorizerMsg.transform(cleanMessage['newMessage'])
#convert message and subject data's vector to array 
vectorSubject = vectorSubject.toarray()
vectorMessage = vectorMessage.toarray()
#use numpy's hstack to join ndarray
x_phishingTrain = np.hstack((vectorSubject, vectorMessage))
y_phishingTrain = dataY['target']

#use logistic regression classifier to train the model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_phishingTrain, y_phishingTrain)

#Phishing Filter Testing
#Test data from real phishing emails

#reading testing data
testData = pd.read_excel(phishingTestPath)
#extracting testing dataset into respective subsections/lists
testDataX = testData.drop('target', axis=1)
testSubjectData = testDataX.drop('message', axis=1)
testMessageData = testDataX.drop('subject', axis=1)
testRawY = testData.drop('subject', axis=1)
testDataY = testRawY.drop('message', axis=1)

#cleaning testing dataset
cleanTestSubject = testSubjectData['subject'].apply(lambda row: cleanText(row))
cleanTestSubject = pd.DataFrame({'newSubject': cleanTestSubject})
cleanTestMessage = testMessageData['message'].apply(lambda row: cleanText(row))
cleanTestMessage = pd.DataFrame({"newMessage": cleanTestMessage})

# Vectorizing testing dataset
vectorTestSub = vectorizerSub.transform(cleanTestSubject['newSubject'])
vectorTestMsg = vectorizerMsg.transform(cleanTestMessage['newMessage'])
vectorTestSub = vectorTestSub.toarray()
vectorTestMsg = vectorTestMsg.toarray()
x_phishingTest = np.hstack((vectorTestSub, vectorTestMsg))
y_phishingTest = testDataY['target']

#predicting phishing email data using phishing filter
from sklearn.metrics import f1_score, accuracy_score
print("Predecting Claflin's phishing email data by Linear Regression:")
#predict using trained lr
predicted = lr.predict(x_phishingTest)
print("The F1 score is: ", f1_score(predicted, y_phishingTest))
print("The accuracy score is: ", accuracy_score(predicted, y_phishingTest))

#print confusion matrix using model's results
from sklearn.metrics import confusion_matrix
import seaborn as sns
cf_matrix = confusion_matrix(y_phishingTest, predicted)
sns.heatmap(cf_matrix, annot=True)
