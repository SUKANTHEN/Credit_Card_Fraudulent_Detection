import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('C:/Users/admin/Downloads/train_20D8GL3.csv', sep=',')
test = pd.read_csv('C:/Users/admin/Downloads/test_O6kKpvt.csv', sep=',')
X = tpot_data.drop('default_payment_next_month', axis=1)
y = tpot_data['default_payment_next_month']

#Average CV score on the training set was: 0.8205357142857143
exported_pipeline = RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.4, min_samples_leaf=18, min_samples_split=16, n_estimators=100)

exported_pipeline.fit(X,y)
results = exported_pipeline.predict(test)



