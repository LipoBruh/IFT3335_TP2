import pandas as pd
import numpy as np 
#models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
#Preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
#vectorizers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#Metrics
from sklearn.metrics import make_scorer, accuracy_score, f1_score
#stacking / super learner
from sklearn.ensemble import StackingClassifier
#
#Global variables
PATH = "../spam.csv"
FILE = "./output_3_1.txt"
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






print("THE TRAINING IS LONG... KILL THE PROCESS IF TOO LONG")
print("RESULTS ALREADY IN OUTPUT.TXT")
print("NEW TRAINING WILL OVERWRITE DATA")





#Documentation on a similar project
#["Jupiter Notebook on a Spam Classifier"]("https://colab.research.google.com/github/tbass134/ml-portfolio/blob/master/_notebooks/2020-11-06-SMS_Spam_Classifier_Demo.ipynb")

#Cleaning up the data (Part 0.1 and 0.2)
df = cleanup(parser(PATH))
X = df["v2"]
y = df["v1"]


### Part 0.3 et 0.4

###Vectorizer parameters

param_grid = {
    'tfidf__max_features': [5000],      
    'tfidf__lowercase': [True],     
    'tfidf__strip_accents': ["unicode"]
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


pipe_LR = Pipeline([
    ("clf",LogisticRegression())
    ])

pipe_SVC = Pipeline([
        ("clf",SVC(probability=True))
    ])

pipe_RF = Pipeline([
    ("clf",RandomForestClassifier())
    ])
#
meta_learner = LogisticRegression()
#
#
#
print("Training Super Learner w/ TF-IDF...")

stack = StackingClassifier(
    estimators=[
        ('lr', pipe_LR),
        ('svc', pipe_SVC),
        ('rf',pipe_RF)
    ],
    final_estimator=meta_learner,
    cv=5,
)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ('stack', stack)
])
grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring=scoring,
    refit='f1'
).fit(X,y)

print("DO NOT QUIT PAST THIS STEP (SAVING SCORES...)")
#
clear_file()
#
append_to_file("\n\n")
append_to_file("F1 & ACCURACY SCOFRES\n")
append_frame_to_file(pd.DataFrame(grid.cv_results_))
append_to_file("\n\n")
append_to_file("Coefficients : \n")

df = pd.DataFrame(grid.cv_results_)
df.to_csv("../output/3_1_superlearner.csv", index=False)
#

#3.2
best_pipeline = grid.best_estimator_
stacking_model = best_pipeline.named_steps['stack']
meta_model = stacking_model.final_estimator_
meta_model_weights = meta_model.coef_
#
append_array_to_file(meta_model_weights)

print('...DONE...')