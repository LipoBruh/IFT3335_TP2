import pandas as pd
import numpy as np 
#models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
#Preprocessing
from sklearn.preprocessing import StandardScaler #for normalizing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
#vectorizers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#Metrics
from sklearn.metrics import make_scorer, accuracy_score, f1_score
#Global variables
PATH = "./spam.csv"
#Helper fx
from parser1 import parser, cleanup


#Documentation on a similar project
#["Jupiter Notebook on a Spam Classifier"]("https://colab.research.google.com/github/tbass134/ml-portfolio/blob/master/_notebooks/2020-11-06-SMS_Spam_Classifier_Demo.ipynb")

#Cleaning up the data (Part 0.1 and 0.2)
df = cleanup(parser(PATH))
X = df["v2"]
y = df["v1"]


### Part 0.3 et 0.4

###Vectorizer parameters
param_grid1 = {
    'vect__max_features': 5000,      
    'vect__lowercase': True,     
    'vect__strip_accents': "unicode"
}
param_grid2 = {
    'tfidf__max_features': 5000,      
    'tfidf__lowercase': True,     
    'tfidf__strip_accents': "unicode"
}
### Scorer
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score, pos_label='spam')
}



### Part 0.5
# cv=5 -> 5 folds
# scoring -> accuracy
# 
print("Training LR...")
pipe_LR1 = Pipeline([
    ("model",LogisticRegression()),
    ("vect",CountVectorizer())
    ])
pipe_LR2 = Pipeline([
    ("model",LogisticRegression()),
    ("tfidf", TfidfVectorizer())
    ])
#
print("Training LR BOF...")
model_LR1 = GridSearchCV(
    estimator=pipe_LR1,
    param_grid=param_grid1,
    cv=5,
    scoring=scoring,
    ).fit(X,y)
#
print("Training LR IDF...")
model_LR2 = GridSearchCV(
    estimator=pipe_LR2,
    param_grid=param_grid2,
    cv=5,
    scoring=scoring,
    ).fit(X,y)
#
#
#
#
print("Training RF...")
#Random Forest
pipe_RF1 = Pipeline([
    ("model",RandomForestClassifier()),
    ("vect",CountVectorizer())
    ])
pipe_RF2 = Pipeline([
    ("model",RandomForestClassifier()),
    ("tfidf", TfidfVectorizer())
    ])
#
#
#
print("Training RF BOF...")
model_RF1 = GridSearchCV(
    estimator=pipe_RF1,
    param_grid=param_grid1,
    cv=5,
    scoring=scoring,
    ).fit(X,y)
#
print("Training RF IDF...")
model_RF2 = GridSearchCV(
    estimator=pipe_RF2,
    param_grid=param_grid2,
    cv=5,
    scoring=scoring,
    ).fit(X,y)

#
#
#
#
print("Training MLP...")
pipe_MLP1 = Pipeline([
    ("model",MLPClassifier()),
    ("vect",CountVectorizer())
    ])
pipe_MLP2 = Pipeline([
    ("model",MLPClassifier()),
    ("tfidf", TfidfVectorizer())
    ])
#
print("Training MLP BOF...")
model_MLP1 = GridSearchCV(
    estimator=pipe_MLP1,
    param_grid=param_grid1,
    cv=5,
    scoring=scoring,
    ).fit(X,y)
#
print("Training MLP IDF...")
model_MLP2 = GridSearchCV(
    estimator=pipe_MLP2,
    param_grid=param_grid2,
    cv=5,
    scoring=scoring,
    ).fit(X,y)

