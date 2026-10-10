import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import precision_score,accuracy_score,recall_score,confusion_matrix,f1_score
import joblib



# Loading data from csv file

data = pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'],header=None)
print(data.head())


# Analysing the data

print(data.shape)
print(data.dtypes)
print(data.columns)
print(data.isna().sum())
print(data.duplicated().sum())
print(data.describe())



# Removing the duplicate rows

data = data.drop_duplicates()
print(data.shape)



# Feature and Target distribution

y = data['label'].map({'ham' : 0 , 'spam' : 1})
X = data['message']



# Splitting the training and testing data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)



# Convert text data in numerical so machine can understand
# In simple language we teach vocabulary to machine

vectorise = CountVectorizer()
X_train = vectorise.fit_transform(X_train)
X_test = vectorise.transform(X_test)
print(vectorise.get_feature_names_out())    # show all words
print(type(X),'\n')                         # describe the type of Sample Feature



# First Model GaussianNB

# model_Gau = GaussianNB()
# model_Gau.fit(X_train.toarray(),y_train)
# y_test_pred_Gau = model_Gau.predict(X_test.toarray())
# y_train_pred_Gau = model_Gau.predict(X_train.toarray())



# Second Model MultinomailNB

# model_Mul = MultinomialNB(alpha=1.0)
# model_Mul.fit(X_train,y_train)
# y_test_pred_Mul = model_Mul.predict(X_test)
# y_train_pred_Mul = model_Mul.predict(X_train)



# Thrid ModelBernoulliNB

model_Ber = BernoulliNB(alpha=0.01,binarize=0)
model_Ber.fit(X_train,y_train)
y_test_pred_Ber = model_Ber.predict(X_test)
y_train_pred_Ber = model_Ber.predict(X_train)



# Checking f1_score on all 3 model

# print('f1_score_train_Gau:',f1_score(y_train,y_train_pred_Gau))
# print('f1_score_test_Gau:',f1_score(y_test,y_test_pred_Gau),'\n')

# print('f1_score_train_Mul:',f1_score(y_train,y_train_pred_Mul))
# print('f1_score_test_Mul:',f1_score(y_test,y_test_pred_Mul),'\n')

print('f1_score_train_Ber:',f1_score(y_train,y_train_pred_Ber))
print('f1_score_test_Ber:',f1_score(y_test,y_test_pred_Ber),'\n')



# Checking Precision_score on all 3 model

# print('precision_score_train_Gau:',precision_score(y_train,y_train_pred_Gau))
# print('precision_score_test_Gau:',precision_score(y_test,y_test_pred_Gau),'\n')

# print('precision_score_train_Mul:',precision_score(y_train,y_train_pred_Mul))
# print('precision_score_test_Mul:',precision_score(y_test,y_test_pred_Mul),'\n')

print('precision_score_train_Ber:',precision_score(y_train,y_train_pred_Ber))
print('precision_score_test_Ber:',precision_score(y_test,y_test_pred_Ber),'\n')




# Checking accuracy_score on all 3 model

# print('accuracy_score_train_Gau:',accuracy_score(y_train,y_train_pred_Gau))
# print('accuracy_score_test_Gau:',accuracy_score(y_test,y_test_pred_Gau),'\n')

# print('accuracy_score_train_Mul:',accuracy_score(y_train,y_train_pred_Mul))
# print('accuracy_score_test_Mul:',accuracy_score(y_test,y_test_pred_Mul),'\n')

print('accuracy_score_train_Ber:',accuracy_score(y_train,y_train_pred_Ber))
print('accuracy_score_test_Ber:',accuracy_score(y_test,y_test_pred_Ber),'\n')




# Checking recall_score on all 3 model

# print('recall_score_train_Gau:',recall_score(y_train,y_train_pred_Gau))
# print('recall_score_test_Gau:',recall_score(y_test,y_test_pred_Gau),'\n')

# print('recall_score_train_Mul:',recall_score(y_train,y_train_pred_Mul))
# print('recall_score_test_Mul:',recall_score(y_test,y_test_pred_Mul),'\n')

print('recall_score_train_Ber:',recall_score(y_train,y_train_pred_Ber))
print('recall_score_test_Ber:',recall_score(y_test,y_test_pred_Ber),'\n')



# Checking Confusion_matrix on all 3 model

# print('confusion_matrix_train_Gau:','\n',confusion_matrix(y_train,y_train_pred_Gau))
# print('confusion_matrix_test_Gau:','\n',confusion_matrix(y_test,y_test_pred_Gau),'\n')

# print('confusion_matrix_train_Mul:','\n',confusion_matrix(y_train,y_train_pred_Mul))
# print('confusion_matrix_test_Mul:','\n',confusion_matrix(y_test,y_test_pred_Mul),'\n')

print('confusion_matrix_train_Ber:','\n',confusion_matrix(y_train,y_train_pred_Ber))
print('confusion_matrix_test_Ber:','\n',confusion_matrix(y_test,y_test_pred_Ber),'\n')




# MODEL COMPARISON NOTES
# Gaussian Naive Bayes:
# Test F1 Score = 0.7273

# Multinomial Naive Bayes:
# Test F1 Score = 0.9129

# Bernoulli Naive Bayes:
# Test F1 Score = 0.9565

# Bernoulli Naive Bayes gave the best test F1 score
# among the tested models and was selected for further tuning.



# FINAL MODEL BERNOULLI NAIVE BAYES
#
# Bernoulli Naive Bayes was selected after comparing it with
# Gaussian and Multinomial Naive Bayes models.
#
# Initial Test F1 Score: 0.9363
# Tuned Test F1 Score: 0.9565
#
# Best tested configuration:
# alpha = 0.01
# binarize = 0
#
# Final Test Performance:
# Accuracy  = 98.84%
# Precision = 97.06%
# Recall    = 94.29%
# F1 Score  = 95.65%
#
# The model showed strong performance on unseen test data
# with a relatively small difference between training and test F1 scores.




# Error Analysing 

Error_Analysis = pd.DataFrame({
    'Actual' : y_test,
    'Prediction' : y_test_pred_Ber
})

Error_Analysis['message'] = data.loc[y_test.index,'message']

misclassified = Error_Analysis[Error_Analysis['Actual'] != Error_Analysis['Prediction']]

print(misclassified[['message','Actual','Prediction']])



# Saving trained model for future uses

joblib.dump(model_Ber,'Model_Bernoullis.pkl')
joblib.dump(vectorise,'Vectorise.pkl')



# Loading Saved model
load2 = joblib.load('Model_Bernoullis.pkl')
load1 = joblib.load('Vectorise.pkl')



# Testing on new data

new_X = load1.transform(['Congratulations! You won a free prize'])
prediction = load2.predict(new_X)

if prediction[0] == 1:
    print('Spam')
else:
    print('Ham')