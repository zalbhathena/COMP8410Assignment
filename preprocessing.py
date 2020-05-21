import pandas as pd
from pandas.api.types import is_string_dtype

redundant_columns = ['Q11','Q12','Q14','Q15','Q16','Q18']

def getPreprocessedFile():
    df = pd.read_csv('3425_data.csv',sep=',', engine='python')
    df = df.drop(['srcid', 'Mode', "total_time_taken",'neutral_count','disagree_count','agree_count','very_count','just_count','vmj_count', 'sDevType'], axis=1) 
    for i, col in enumerate(df.columns):
        if is_string_dtype(df[col]):
            df.iloc[:, i] = df.iloc[:, i].str.replace('"', '') 
            #df.iloc[:, i] = df.iloc[:, i].str.replace('-', '') 
            df.iloc[:, i] = df.iloc[:, i].str.replace('NaN', '0')

    df.fillna(0, inplace=True)
    for column in df: 
        df[column] = pd.to_numeric(df[column])
    return df
