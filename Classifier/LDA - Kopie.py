from pathlib import Path
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd

import os 
import itertools
import re
import math
import csv
import random
from sklearn import tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from scipy.fftpack import fftn, ifftn, fft, ifft
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import signal
import pickle
from sklearn.externals import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import warnings

warnings.filterwarnings('ignore') 
df = pd.read_csv("Features_all.csv")

def split_data_per_session(session_id):
    temp_df = df[(df.sessionID == session_id)]
    data = temp_df.copy()
    data = shuffle(data)
    y = data['modeID']
    data.drop(labels=['userID','uttID', 'sessionID','modeID'], axis=1, inplace=True)
    return train_test_split(data, y, test_size=0.2)

def split_data():
    temp_df = df
    data = temp_df.copy()
    data = shuffle(data)
    y = data['modeID']
    data.drop(labels=['userID','uttID', 'sessionID','modeID'], axis=1, inplace=True)
    return train_test_split(data, y, test_size=0.2)

def split_data_per_user(user_id):
    temp_df = df[(df.userID == user_id)]
    data = temp_df.copy()
    data = shuffle(data)
    y = data['modeID']
    data.drop(labels=['userID','uttID', 'sessionID','modeID'], axis=1, inplace=True)
    return train_test_split(data, y, test_size=0.2)
    
def lda_all_users_separately():
    user_ids = np.unique(df.userID)
    for user_id in user_ids:
        X_train, X_test, y_train, y_test = split_data_per_user(user_id)
        k_clf = LinearDiscriminantAnalysis()
        parameter_grid = {}
        clf = GridSearchCV(k_clf, parameter_grid, cv=10)
        clf.fit(X_train, y_train)
        print("Accuracy for user " + str(user_id) + " is " + str(clf.score(X_test, y_test)))

def lda_all():
    X_train, X_test, y_train, y_test = split_data()
    k_clf = LinearDiscriminantAnalysis()
    parameter_grid = {}
    clf = GridSearchCV(k_clf, parameter_grid, cv=10)
    clf.fit(X_train, y_train)
    print("Accuracy for all users is "+str(clf.score(X_test, y_test)))
    
def lda_all_sessions_separately():
    sessionIDs = np.unique(df.sessionID)
    for sessionID in sessionIDs:
        X_train, X_test, y_train, y_test = split_data_per_session(sessionID)
        k_clf = LinearDiscriminantAnalysis()
        parameter_grid = {}
        clf = GridSearchCV(k_clf, parameter_grid, cv=10)
        clf.fit(X_train, y_train)
        print("Accuracy for session " + str(sessionID) + " is " + str(clf.score(X_test, y_test)))

# lda_all()
# lda_all_users_separately()
# lda_all_sessions_separately