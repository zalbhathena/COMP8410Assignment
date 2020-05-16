import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from pandas.api.types import is_string_dtype

import itertools

df = pd.read_csv('3425_data.csv',sep=',', engine='python')
#df = df.drop(['srcid', 'Mode', 'sDevType','p_gender_sdc','p_age_group_sdc','p_education_sdc','total_time_taken','neutral_count','disagree_count','agree_count','very_count','just_count','vmj_count','opinionated','undecided_voter'], axis=1)
#df = df.drop(['Q1','Q2','Q5a','Q5f','Q6c','Q7a','Q7c','Q8a','Q8d','Q10a','Q10d','Q11','Q11a','Q12','Q14','Q15','Q16','Q18'], axis=1)
df = df.drop(['srcid', 'Mode', "total_time_taken",'neutral_count','disagree_count','agree_count','very_count','just_count','vmj_count'], axis=1)
for i, col in enumerate(df.columns):
    if is_string_dtype(df[col]):
        df.iloc[:, i] = df.iloc[:, i].str.replace('"', '')
        #df.iloc[:, i] = df.iloc[:, i].str.replace('-', '')
        df.iloc[:, i] = df.iloc[:, i].str.replace('NaN', '0')

df.fillna(0, inplace=True)
for column in df:
    df[column] = pd.to_numeric(df[column])

for column in df:
    df[column] = pd.Categorical(df[column])
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

    accuracy = accuracy_score(y_test, y_pred)*100
    important_neg_features= pd.DataFrame(data=np.transpose(model.feature_log_prob_).astype("float32"),index=x_train.columns)[0].sort_values()
    important_pos_features= pd.DataFrame(data=np.transpose(model.feature_log_prob_).astype("float32"),index=x_train.columns)[1].sort_values()
    print(col, accuracy)
