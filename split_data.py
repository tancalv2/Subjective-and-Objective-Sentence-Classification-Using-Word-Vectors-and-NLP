"""
    3.1 Create train/validation/test splits

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


# ================================ TRAINING, VALIDATION AND TEST DATA SPLIT ======================================== #

seed = 0

# 3.1 YOUR CODE HERE

data = pd.read_csv('./data/data.tsv', sep='\t')

# SPLIT OBJECTIVE FROM SUBJECTIVE
obj = data[data['label'] == 0]
sub = data[data['label'] == 1]

# SPLIT DATASETS
obj_tv, obj_test = train_test_split(obj, test_size=0.2, random_state=seed)       # 20% in test set
obj_train, obj_val = train_test_split(obj_tv, test_size=0.2, random_state=seed)  # 1/5 of 80% in val set

sub_tv, sub_test = train_test_split(sub, test_size=0.2, random_state=seed)       # 20% in test set
sub_train, sub_val = train_test_split(sub_tv, test_size=0.2, random_state=seed)  # 1/5 of 80% in val set

# RECONCATENATE SUB AND OBJ
train = obj_train.append(sub_train)
val = obj_val.append(sub_val)
test = obj_test.append(sub_test)

# Counts
print(train["label"].value_counts())
print(val["label"].value_counts())
print(test["label"].value_counts())
# 1    3200
# 0    3200
# Name: label, dtype: int64
# 1    800
# 0    800
# Name: label, dtype: int64
# 1    1000
# 0    1000
# Name: label, dtype: int64

# SAVE
pd.DataFrame.to_csv(train, './data/train.tsv', sep='\t', index=False)
pd.DataFrame.to_csv(val, './data/val.tsv', sep='\t', index=False)
pd.DataFrame.to_csv(test, './data/test.tsv', sep='\t', index=False)

######
