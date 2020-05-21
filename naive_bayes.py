import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import csv

import preprocessing

df = preprocessing.getPreprocessedFile()


dummy_df = pd.DataFrame()
for column in df:
    if column == "Q4":
        continue
    dummies = pd.get_dummies(df[column])
    for dummy_col in dummies:
        dummy_df[str(column) + "_" + str(dummy_col)] = dummies[dummy_col]

dummy_q4 = pd.get_dummies(df["Q4"])

for col in dummy_q4:
    # naive bayes trained to predict politcal party
    #x = df[list(combo)]
    x = dummy_df
    y = dummy_q4[col]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=43)

    model = BernoulliNB()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    #print(y_pred)

    accuracy = accuracy_score(y_test, y_pred)*100
    important_neg_features= pd.DataFrame(data=np.transpose(model.feature_log_prob_).astype("float32"),index=x_train.columns)[0].sort_values()
    important_pos_features= pd.DataFrame(data=np.transpose(model.feature_log_prob_).astype("float32"),index=x_train.columns)[1].sort_values()
    print(col, accuracy)
    print(confusion)
    print(report)
