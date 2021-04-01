"""
Constructs one of the initial models in this series of experiments. In the previous experiments, RandomForest model
had the best performance on the feature extracted datasets. This model was selected and hyperparameters were tuned
on it. This models is saved as ML_model.joblib.
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
import pickle
from joblib import dump, load

# Reads in the labels file
df = pd.read_pickle('Datasets/Labels_mac.pkl')

df = df[:12502]

# Drops the non-number values
df = df.drop(['Original File'], axis=1)

# Shuffles the dataset
df = shuffle(df)

# Balances the classes
def bal_class(df):

    # Gets the minority class number
    labs = list(set(df['Labels']))
    lens = []
    for l in labs:
        a_len = len(df[df['Labels'] == l])
        lens.append(a_len)

    min_val = np.min(lens)

    # Creates a dataframe for each label that is the size of the minority class
    dfs = []
    for l in labs:
        df_w = df[df['Labels'] == l]
        dfs.append(df_w[:min_val])

    dff = pd.concat(dfs)
    return dff

# df = bal_class(df)

# Splits the X and y data
X = df.iloc[:,:-1]
y = df['Labels']
y = y.ravel()

# Performs the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13411)

model = RandomForestClassifier(n_estimators=1600,
                               min_samples_split= 2,
                               min_samples_leaf= 4,
                               max_features='sqrt',
                               max_depth=10,
                               bootstrap= True,
                               verbose=1)

scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
clf = model.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

probs = [float(x[1]) for x in y_prob]

dff = pd.DataFrame({
    'Predictions': y_pred,
    "Probability": probs,
    'Correct': y_test
})

rep = classification_report(y_test,y_pred)
print(rep)
def ratio_maker (pred, cor):
    if pred == cor:
        return 1
    else:
        return 0

def binner (prob):
    bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    obj = []
    while len(obj) == 0:
        for b in bins:
            if prob <= b:
                obj.append(b)

    return obj[0]

dff['Ratio'] = dff.apply(lambda x: ratio_maker(x['Predictions'], x['Correct']), axis=1)
dff['Bin'] = dff['Probability'].apply(binner)

sns.countplot(x='Bin', hue='Ratio', data=dff)
plt.show()

def run_exps(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
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
        ('RF', RandomForestClassifier(verbose=1, type='prob')),
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

def rf_rd_search():
    from sklearn.model_selection import RandomizedSearchCV
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=1341, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)

    best = rf_random.best_estimator_
    print(rf_random.best_params_)
    predictions = best.predict(X_test)
    res = confusion_matrix(y_test, predictions)
    print(res)


dump(model, 'ML_model.joblib')







