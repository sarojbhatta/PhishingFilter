# -*- coding: utf-8 -*-
"""Final Spam Filter Training Testing.ipynb
Written and generated from https://colab.research.google.com/
Written by: Saroj Bhatta, CS '21, Alice Carson Tisdale Honors College, Claflin University
"""

#Spam Detection Filter Development

spamTrainPath = "/content/SpamTrainData.xlsx"
spamTestPath = "/content/spamPhishingTestData_Claflin.xlsx"

#importing important libraries
import pandas as pd

#importing training data and separating
data = pd.read_excel(spamTrainPath)
data = data.drop('sn', axis=1)
y_data = data.drop('message', axis=1)
x_data = data.drop('target', axis=1)

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

#clean data using cleanText function
xClean = x_data['message'].apply(lambda row: cleanText(row))
xClean = pd.DataFrame({"newMessage": xClean})

#import and use vectorizer to fit and transform cleaned data
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(xClean['newMessage'])
vector = vectorizer.transform(xClean['newMessage'])
x_value = vector.toarray()
y_value = y_data['target']

#train-test data split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_value, y_value, test_size=0.25, random_state=42)

#use logistic regression classifier to train the model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

#predict model's accuracy from splitted test-data
from sklearn.metrics import f1_score, accuracy_score
print("\nPredecting model's accuracy:")
predictedTest = lr.predict(x_test)
print("The F1 score is: ", f1_score(predictedTest, y_test))
print("The accuracy score is: ", accuracy_score(predictedTest, y_test))

#Spam Filter Testing Method
#Test data from real phishing emails

#importing testing data and cleaning
claflinData = pd.read_excel(spamTestPath)
tempClaflinData = claflinData.drop('subject', axis=1)
x_claflinData = tempClaflinData.drop('target', axis=1)
y_claflinData = tempClaflinData.drop('message', axis=1)
xClaflinClean = x_claflinData['message'].apply(lambda row: cleanText(row))
xClaflinClean = pd.DataFrame({"newbody": xClaflinClean})

#use vectorizer to transform testing data
vectorClaflin = vectorizer.transform(xClaflinClean['newbody'])
x_ClaflinValue = vectorClaflin.toarray()
y_ClaflinValue = y_claflinData['target']

#predicting phishing email data using spam filter
print("Predecting Claflin's email data through Spam Filter using Linear Regression: \n")
predictClaflin = lr.predict(x_ClaflinValue)
print("The F1 score is: ", f1_score(predictClaflin, y_ClaflinValue))
print("The accuracy score is: ", accuracy_score(predictClaflin, y_ClaflinValue))

#print confusion matrix using model's results
from sklearn.metrics import confusion_matrix
import seaborn as sns
cf_matrix = confusion_matrix(y_ClaflinValue, predictClaflin)
sns.heatmap(cf_matrix, annot=True)
