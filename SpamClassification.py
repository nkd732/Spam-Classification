import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score


def preprocess():
    data_frame = pd.read_csv("enron_dataset.csv", sep = ',')

    emails_total = data_frame.to_numpy()

    np.random.shuffle(emails_total)

    emails = emails_total[0:12000]         

    characters = ['~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '{', '['
            '}', ']', '|', '\\',':', ';', '"', "'", "<", ",", ">", ".", "?", "/"]

    suffixes = ['ing', 'es', 'ly', 'ful', 'ic', 'ous', 'ism', 'ise', 'ize', 'ion', 'ment', 'en']

    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    for i in range(len(emails)):

        if pd.isnull(emails[i, 1]):
            emails[i, 1] = ''

        if pd.isnull(emails[i, 2]):
            emails[i, 2] = '' 

        emails[i, 1].lower()
        emails[i, 2].lower()
        
        for c in characters:
            emails[i, 1] = emails[i, 1].replace(c, '')
            emails[i, 2] = emails[i, 2].replace(c, '')

        for s in suffixes:
            emails[i, 1] = emails[i, 1].replace(s, '')
            emails[i, 2] = emails[i, 2].replace(s, '')
        
        for n in numbers:
            emails[i, 1] = emails[i, 1].replace(n, '0')
            emails[i, 2] = emails[i, 2].replace(n, '0')

    np.save('emails.npy', emails)
    
    print('Preprocessing completed!')

    return 

def vectorize():

    emails = np.load('emails.npy', allow_pickle = True)
    documents = [[]]
    index = 0

    vocabulary = {}

    for i in range(len(emails)):
        subject = emails[i, 1].split()
        body = emails[i, 2].split()
        words = subject + body
        documents.append(words)

    documents.pop(0)

    for i in range(len(documents[0:10000])):
        for w in documents[i]:
            if w not in vocabulary:
                vocabulary[w] = index
                index += 1
            
    count = {}

    for v in vocabulary.keys():
        count[v] = 0.0

    for d in documents[0:10000]:
        d = list(set(d))
        for w in d:
            count[w] += 1

    invDocFreq = {}

    for w in count.keys():
        invDocFreq[w] = 1.0 + np.log((1.0 + len(documents[0:10000]))/(1.0 + count[w]))

    np.save('vocabulary.npy', vocabulary)
    np.save('invDocFreq.npy', invDocFreq)
    print('the invDocFreq and vocabulary have been saved to invDocFreq.npy and vocabulary.npy')

    X = np.zeros((len(documents), len(vocabulary)), dtype = 'float16')

    for i in range(len(documents)):
        tf_idf = np.zeros(len(vocabulary), dtype = 'float16')
        noOfWords = 0
        if documents[i] != []:
            for w in documents[i]:
                if w in vocabulary:
                    tf_idf[vocabulary[w]] += invDocFreq[w]
                    noOfWords += 1
            tf_idf /= noOfWords
        X[i, :] = tf_idf

    X_train = X[0:10000]
    X_test = X[10000:12000]

    y = np.zeros(len(emails), dtype = 'int32')

    for i in range(len(emails)):
        if emails[i, 3] == 'ham':
            y[i] = 0
        else:
            y[i] = 1

    y_train = y[0:10000]
    y_test = y[10000:12000]

    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    print('Vectorization of training data & test data is complete')
    
    return 

def train():    
    X_train = np.load('X_train.npy', allow_pickle = True)
    y_train = np.load('y_train.npy', allow_pickle = True)
    
    print('Support Vector Classifier')
    classifier = SVC(kernel = 'linear', C = 6.0)
    print('Training is in progress')
    classifier.fit(X_train, y_train)
    print('Training has been completed')
    joblib.dump(classifier, 'svc.sav')
    print('The model has been saved to svc.sav')
    
    print('Multinomial Naive Bayes')
    classifier = MultinomialNB()
    print('Training is in progress')
    classifier.fit(X_train, y_train)
    print('Training has been completed')
    joblib.dump(classifier, 'multinomialNB.sav')
    print('The model has been saved to multinomialNB.sav')
    
    print('Bernoulli Naive Bayes')
    classifier = BernoulliNB()
    print('Training is in progress')
    classifier.fit(X_train, y_train)
    print('Training has been completed')
    joblib.dump(classifier, 'bernoulliNB.sav')
    print('The model has been saved to bernoulliNB.sav')

    print('Logistic Regression')
    classifier = LogisticRegression()
    print('Training is in progress')
    classifier.fit(X_train, y_train)
    print('Training has been completed')
    joblib.dump(classifier, 'logisticRegression.sav')
    print('The model has been saved to logisticRegression.sav')
 
    return

def predict():
    X_test = np.load('X_test.npy', allow_pickle = True)
    y_test = np.load('y_test.npy', allow_pickle = True)

    print('Loading models')
    svcModel = joblib.load('svc.sav')
    mnbModel = joblib.load('multinomialNB.sav')
    bnbModel = joblib.load('bernoulliNB.sav')
    logRegModel = joblib.load('logisticRegression.sav')
     
    print('Making predictions')
    svcPred = svcModel.predict(X_test)
    mnbPred = mnbModel.predict(X_test)
    bnbPred = bnbModel.predict(X_test)
    logRegPred = logRegModel.predict(X_test)

    print('Calculating accuracy')
    svcScore = accuracy_score(svcPred, y_test)
    mnbScore = accuracy_score(mnbPred, y_test)
    bnbScore = accuracy_score(bnbPred, y_test)
    logRegScore = accuracy_score(logRegPred, y_test)

    print('The accuracy score for Support Vector Classification is : ', svcScore)
    print('The accuracy score for Multinomial Naive Bayes Classifier is :', mnbScore)
    print('The accuracy score for Bernoulli Naive Bayes Classifier is :', bnbScore)
    print('The accuracy score for Logistic Regression is :', logRegScore)
    
    return 

# Main Program ---------------------------------------------------------------------------------------------------------------------------

preprocess()
vectorize()
train()
predict()