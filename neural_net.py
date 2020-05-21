import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np



df = preprocessing.getPreprocessedFile()


dummy_df = pd.DataFrame()
for column in df: 
    if column == "Q4":
        continue
    dummies = pd.get_dummies(df[column])
    for dummy_col in dummies:
        dummy_df[str(column) + "_" + str(dummy_col)] = dummies[dummy_col]

dummy_q4 = pd.get_dummies(df["Q4"])

res = []
for col in dummy_q4:
    x = dummy_df
    y = dummy_q4[col]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=43)

    # Initialize the constructor
    model = Sequential()

    # Add an input layer 
    model.add(Dense(162, activation='relu', input_shape=(162,)))

    # Add one hidden layer 
    model.add(Dense(81, activation='relu'))

    # Add an output layer 
    model.add(Dense(1, activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        name="Adam"
    )
    model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

    model.fit(x_train, y_train,epochs=200, batch_size=1, verbose=0)

    y_pred = model.predict(x_test)
    score = model.evaluate(x_test, y_test,verbose=1)
    y_pred_bool = pd.DataFrame(y_pred).round(0).astype(int)[0]
    confusion = confusion_matrix(y_test, y_pred_bool)
    
    report = classification_report(y_test, y_pred_bool)
    res.append((col, score, report, confusion))
print("\n".join([str(x[:2]) for x in res]))
print("\n".join([str(x[2]) for x in res]))
print("\n".join([str(x[3]) for x in res]))
