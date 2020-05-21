import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

import preprocessing

df = preprocessing.getPreprocessedFile()

fig, axes = plt.subplots(4, 6)
fig.autolayout = True

count = 0
for col in df:
    df[col].value_counts().plot(kind='bar', title=col, ax=axes[count//6,count%6])
    if col == "Q2":
        axes[count//6,count%6].axis('off')
    count += 1
plt.subplots_adjust(top = 0.95, bottom=0.07, hspace=0.36, wspace=0.34)
plt.show()
