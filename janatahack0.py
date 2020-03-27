import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('C:/Users/admin/Downloads/train_20D8GL3.csv', sep=',')
features = tpot_data.drop('default_payment_next_month', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['default_payment_next_month'], random_state=None)

# Average CV score on the training set was: 0.8178571428571428
exported_pipeline = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.7000000000000001, min_samples_leaf=15, min_samples_split=5, n_estimators=100)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print('The accuracy of the model is {}, AUC score of the model is {}'.format(accuracy_score(results,testing_target),roc_auc_score(results,testing_target)))
print('The confusion matrix is:', confusion_matrix(results,testing_target))

