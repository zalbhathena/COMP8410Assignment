import preprocessing
from scipy.stats import chisquare
import scipy.stats as ss
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dython.nominal import conditional_entropy
from collections import Counter

df = preprocessing.getPreprocessedFile()

df = df.apply(lambda x : pd.factorize(x)[0])+1

chi_sq = pd.DataFrame([chisquare(df[x].values,f_exp=df.values.T,axis=1)[0] for x in df])
print(chi_sq)

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

cramers_v_list = []
theils_u_list = []
for col1 in df:
    cramer_row = []
    theils_row = []
    for col2 in df:
        cramer_row.append(cramers_v(df[col1],df[col2]))
        theils_row.append(theils_u(df[col1],df[col2]))
    cramers_v_list.append(cramer_row)
    theils_u_list.append(theils_row)
cramers_v_df = pd.DataFrame(cramers_v_list, index=df.columns, columns=df.columns)
theils_u_df = pd.DataFrame(theils_u_list, index=df.columns, columns=df.columns)

#sns.heatmap(cramers_v_df, annot=True)
#plt.show()
sns.heatmap(theils_u_df, annot=True)
plt.title("Theil's U")
plt.show()
