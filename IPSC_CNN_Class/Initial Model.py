"""
The initial architecture used with the CNN. Eliza tested some other architectures for CNNs and will have more
information about them.
"""

# Imports
import pandas as pd
import numpy as np
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

# Random state
rand_stat = 13411
# Reads in the labels file
print('Reading')
df = pd.read_csv('../Datasets/CNN_Data_Subset.csv')
print('Done')

# Drops the non-number values
df = df.drop(['Original File','Time','Unnamed: 0', 'Unnamed: 0.1'], axis=1)

# Shuffles the dataset
df = shuffle(df, random_state=rand_stat)

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

df = bal_class(df)

# Splits the X and y data
X = df.drop(['Labels'], axis=1)
X = df.values
y = [np.asarray(i) for i in df['Labels']]
y = np.asarray(y)
y = y.ravel()
print(type(y))

# Performs the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_stat)

X_train_reshape = []
for i in X_train:
    out = np.reshape(i, (input_shape,1,))
    X_train_reshape.append(out)


model = Sequential()

model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(input_shape, 1, )))
# model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_size=3))
model.add(Flatten())
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # Try softmax later

model.summary()
print(len(X_train_reshape))
print(len(y_train))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', keras.metrics.AUC()])
model.fit(x=X_train_reshape, y=y_train, batch_size=300, epochs=10)

preds, acc, AUC = model.evaluate(X_test)
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))
print(acc)
print(AUC)





