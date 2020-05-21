import preprocessing
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import re

df = preprocessing.getPreprocessedFile()
dummy_df = pd.DataFrame()

for column in df:
    dummies = pd.get_dummies(df[column])
    for dummy_col in dummies:
        if dummy_col == 0:
            continue
        dummy_df[str(column) + "_" + str(dummy_col)] = dummies[dummy_col]

frequent_itemsets = apriori(dummy_df, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0)
rules.sort_values(["lift"])
rule_tups = []
for _,rule in rules.iterrows():
    skip = False
    for ant in rule["antecedents"]:
        ant_val = re.split("[a-z]*_", ant)[0]
        for con in rule["consequents"]:
            con_val = re.split("[a-z]*_", con)[0]
            if ant_val == con_val:
                skip = True
    if len(rule["consequents"]) > 1:
        skip = True
    keep = False
    for con in rule["consequents"]:
        if con.startswith("p_age_group"):
            keep = True
        if con.startswith("p_gender"):
            keep = True
        if con.startswith("p_education"):
            keep = True
        if con.startswith("Q4"):
            keep = True
    if not skip and keep:
        rule_tups.append((rule["antecedents"], rule["consequents"], rule["confidence"], rule["lift"]))

rule_tups.sort(key=lambda x:x[3])
print("\n".join([str(x) for x in rule_tups]))

