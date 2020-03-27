import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('C:/Users/admin/Downloads/train_20D8GL3.csv', sep=',')
features = tpot_data.drop('default_payment_next_month', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['default_payment_next_month'], random_state=None)

# Average CV score on the training set was: 0.8175000000000001
model = XGBClassifier(learning_rate=0.5, max_depth=1, min_child_weight=12, n_estimators=100, nthread=1, subsample=0.45)

model.fit(training_features, training_target)
results = model.predict(testing_features)
# Evaluation metrics
print('The accuracy of the model is {}, AUC score of the model is {}'.format(accuracy_score(results,testing_target),roc_auc_score(results,testing_target)))
print('The confusion matrix is:', confusion_matrix(results,testing_target))

