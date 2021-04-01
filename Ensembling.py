"""
This file contains the code I put together to test various ML models on the feature extracted data. If you add some
other models to the list of models under run_exps() you should be able to test them and print out some comparison
metrics. I pulled bits and peices of this from online I think, so some of it is messy. I never thought anyone else
look at it. Enjoy.
"""

# Imports
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow
import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, MaxPooling1D, Conv1D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.signal import find_peaks
from scipy import signal
import pyabf
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier


# Import File
df = pd.read_excel('/Users/jamesbrundage/Desktop/Ensemble_Predictions_11.xlsx', sheet_name='Sheet2')

# Shuffles the dataset
df = shuffle(df)

# Balances the classes
def bal_class(df):

    # Gets the minority class number
    labs = list(set(df['Correct']))
    lens = []
    for l in labs:
        a_len = len(df[df['Correct'] == l])
        lens.append(a_len)

    min_val = np.min(lens)

    # Creates a dataframe for each label that is the size of the minority class
    dfs = []
    for l in labs:
        df_w = df[df['Correct'] == l]
        dfs.append(df_w[:min_val])

    dff = pd.concat(dfs)
    return dff

df = bal_class(df)


# Splits the X and y data
X = df.iloc[:,:-1]
y = df['Correct']
# y = y.ravel()

# Performs the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13411)


def run_exps(X_train: pd.DataFrame(), y_train: pd.DataFrame(), X_test: pd.DataFrame(), y_test: pd.DataFrame()):
    '''
    Lightweight script to test many models and find winners
    :param X_train: training split
    :param y_train: training target vector
    :param X_test: test split
    :param y_test: test target vector
    :return: DataFrame of predictions
    '''

    dfs = []

    models = [
        ('LogReg', LogisticRegression()),
        ('RF', RandomForestClassifier(verbose=1)),
        ('KNN', KNeighborsClassifier()),
        ('SVM', SVC(verbose=1)),
        ('GNB', GaussianNB()),
        ('GBC', GradientBoostingClassifier(verbose=1)),
        ('ADABoost',AdaBoostClassifier()),
        ('HGB', HistGradientBoostingClassifier()),
        ('ETC', ExtraTreesClassifier()),
        ('XGB', XGBClassifier(verbosity=1))
    ]
    results = []
    names = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    target_names = ['No Peak', 'Peak']
    for name, model in models:
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(name)
        print(classification_report(y_test, y_pred, target_names=target_names))
        results.append(cv_results)
        names.append(name)
        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
    final = pd.concat(dfs, ignore_index=True)

    return final

def exps ():
    final = run_exps(X_train, y_train, X_test, y_test)

    bootstraps = []
    for model in list(set(final.model.values)):
        model_df = final.loc[final.model == model]
        bootstrap = model_df.sample(n=30, replace=True)
        bootstraps.append(bootstrap)

    bootstrap_df = pd.concat(bootstraps, ignore_index=True)
    results_long = pd.melt(bootstrap_df, id_vars=['model'], var_name='metrics', value_name='values')
    time_metrics = ['fit_time', 'score_time']  # fit time metrics
    ## PERFORMANCE METRICS
    results_long_nofit = results_long.loc[~results_long['metrics'].isin(time_metrics)]  # get df without fit data
    results_long_nofit = results_long_nofit.sort_values(by='values')
    ## TIME METRICS
    results_long_fit = results_long.loc[results_long['metrics'].isin(time_metrics)]  # df with fit data
    results_long_fit = results_long_fit.sort_values(by='values')

    plt.figure(figsize=(20, 12))
    sns.set(font_scale=2.5)
    g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_nofit, palette="Set3")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Comparison of Model by Classification Metric')
    # plt.savefig('./benchmark_models_performance.png',dpi=300)
    plt.show()

exps()
