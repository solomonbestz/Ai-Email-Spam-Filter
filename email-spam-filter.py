# An email spam detection program using python, checks if the mail sent is spam(1) or not (0)

#Using python's libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string



data_frame = pd.read_csv('spam_ham_dataset.csv')

# print(data_frame.head(5)) #print out 5 rows of data from the dataset
# print(data_frame.shape) #get the number of rows and column in our dataset
# print(data_frame.columns) #get the columns in our dataset

# data_frame.drop_duplicates(inplace=True) #Returns none because we have no duplicate

#Number of missing(NAN, NaN, na) data for each column

# data_frame.isnull().sum() #Sample dataset doesn't have any missing data

# print(nltk.download('stopwords')) #Downloading the stopwords package

#Function to process the text
def process_text(text):
    #Remove punctuation, remove stopwords(useless words in data), then return a list of clean text words

    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = "".join(no_punc)


    clean_words = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

    return clean_words

#Show a list of tokens also called lemmas
# print(data_frame['text'].head().apply(process_text))



#Convert the text to a matrix of token counts
messages_bow = CountVectorizer(analyzer=process_text).fit_transform(data_frame['text'])


#split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(messages_bow, data_frame['label_num'], test_size=0.20, random_state = 0)

#Create and train te naive bayes classifier
# classifier = MultinomialNB().fit(x_train, y_train)

#Print the predictions
# print(classifier.predict(x_train ))

#Print the actual values
# print(y_train.values)


#Evaluate how good our model is on the training data set

# prediction = classifier.predict(x_train)

# Print the Prediction between the actual value(x_train) and the prediction
# print(classification_report(y_train, prediction))

# print()

# print("Confusion Matrix: \n", confusion_matrix(y_train, prediction))

# print()

# Shows how accurate how model predicts a spam mail
# print("Acuracy: \n", accuracy_score(y_train, prediction))


#Print the predictions for test dataset
# print(classifier.predict(x_test))

#Print the actual values
# print(y_test.values)

#Create and train te naive bayes classifier
# classifier = MultinomialNB().fit(x_test, y_test)

#Evaluate how good our model is on the test data set

# prediction = classifier.predict(x_test)
 
# Print the Prediction between the actual value(x_test) and the prediction
# print(classification_report(y_test, prediction))

# print()

# print("Confusion Matrix: \n", confusion_matrix(y_test, prediction))

# print()

# Shows how accurate how model predicts a spam mail
# print("Acuracy: \n", accuracy_score(y_test, prediction))