import pandas as pd
import numpy as np 
#models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
#Preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
#vectorizers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#Metrics
from sklearn.metrics import make_scorer, accuracy_score, f1_score
#Librarie pre entrainee
from sentence_transformers import SentenceTransformer
#Global variables
PATH = "../spam.csv"
FILE = "./output_2_1.txt"
#Helper fx
from parser import parser, cleanup


#
#
#
#
#



def clear_file():
    with open(FILE, 'w'):
        pass  

def append_to_file( text):
    with open(FILE, 'a') as file:
        file.write(text + '\n')

def append_frame_to_file( frame):
    with open(FILE, 'a') as f:
        f.write(frame.to_string(index=False))
        f.write('\n\n')

def append_array_to_file(array):
    with open(FILE, 'a') as f:
        f.write(str(array))
        f.write('\n\n')








#Documentation on a similar project
#["Jupiter Notebook on a Spam Classifier"]("https://colab.research.google.com/github/tbass134/ml-portfolio/blob/master/_notebooks/2020-11-06-SMS_Spam_Classifier_Demo.ipynb")

#Cleaning up the data & embedding (Part 0.1 to 0.3)
#
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#
df = cleanup(parser(PATH))
sentences = df["v2"].values.tolist()
#print(sentences)
embeddings = model.encode(sentences)
#print(embeddings)

print("Embdedding done")


###
print("THE TRAINING IS LONG... KILL THE PROCESS IF TOO LONG")
print("RESULTS ALREADY IN OUTPUT.TXT")
print("NEW TRAINING WILL OVERWRITE DATA")
### Part 2.4 & 2.5
X = embeddings
y = df["v1"]
###Vectorizer parameters
param_grid1 = {}
param_grid2 = {}
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
    ("clf",LogisticRegression())
    ])

model_LR1 = GridSearchCV(
    estimator=pipe_LR1,
    param_grid=param_grid1,
    cv=5,
    scoring=scoring,
    refit='f1',
    ).fit(X,y)

#
#
#
#
print("Training RF...")
#Random Forest
pipe_RF1 = Pipeline([
    ("clf",RandomForestClassifier())
    ])

model_RF1 = GridSearchCV(
    estimator=pipe_RF1,
    param_grid=param_grid1,
    cv=5,
    scoring=scoring,
    refit='f1',
    ).fit(X,y)

#
#
#
#
print("Training MLP...")
pipe_MLP1 = Pipeline([
    ("clf",MLPClassifier())
    ])

print("Training MLP BOF...")
model_MLP1 = GridSearchCV(
    estimator=pipe_MLP1,
    param_grid=param_grid1,
    cv=5,
    scoring=scoring,
    refit='f1',
    ).fit(X,y)
#


print("DO NOT QUIT PAST THIS STEP (SAVING SCORES...)")

clear_file()


append_to_file("\n\n")
append_to_file("F1 & ACCURACY SCOFRES\n")

#1
append_to_file("\n\nLinear Regression Models : \n")
append_frame_to_file(pd.DataFrame(model_LR1.cv_results_))


#2
append_to_file("\n\nRandom Forest Regression Models : \n")
append_frame_to_file(pd.DataFrame(model_RF1.cv_results_))

#3
append_to_file("\n\nMLP Models : \n")
append_frame_to_file(pd.DataFrame(model_MLP1.cv_results_))

print('...DONE...')