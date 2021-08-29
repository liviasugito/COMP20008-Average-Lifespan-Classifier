import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

#load in the data
world = pd.read_csv('world.csv',encoding = 'ISO-8859-1')
life = pd.read_csv('life.csv',encoding = 'ISO-8859-1')

#combine to get feature and label
world_life_join = pd.merge(world, life, how='right', on='Country Code')

#get just the features
world_life_join = world_life_join.replace('..', np.nan)
data = world_life_join.iloc[:,3:-3].astype(float)

#get just the class labels
classlabel = world_life_join.iloc[:,-1]

##randomly select 2/3 of the instances to be training and the rest to be testing
X_train, X_test, y_train, y_test = train_test_split(data,classlabel, train_size=2/3, test_size=1/3, random_state=100)

#impute missing value using median of each column
imp_med = SimpleImputer(missing_values=np.nan, strategy='median')
imp_med = imp_med.fit(X_train)
X_train = imp_med.transform(X_train)
X_test = imp_med.transform(X_test)

#get median imputer value for each column
medians = imp_med.statistics_

#normalise the data to have 0 mean and unit variance using the library functions
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#get mean and variance used for scaling from each column
means = scaler.mean_ 
variances = scaler.var_ 

#combine values to one numpy array and round to 3 decimal places
output = np.column_stack((medians.flatten(),means.flatten(),variances.flatten()))
output = np.round(output,3)

#making task2a.csv
features = pd.DataFrame(list(data), columns=['feature'])
feature_vals = pd.DataFrame(output, columns=['median','mean','variance'])
result = pd.concat([features, feature_vals], axis=1)
result.to_csv (r'task2a.csv', index = False, header=True)

#decision tree with maximum depth=4
dt = DecisionTreeClassifier(random_state=100, max_depth=4)
dt.fit(X_train, y_train)
y_pred_dt=dt.predict(X_test)
print("Accuracy of decision tree:",round(accuracy_score(y_test, y_pred_dt),3))


#knn with neighbor=5
knn5 = neighbors.KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, y_train)

y_pred5=knn5.predict(X_test)

print("Accuracy of k-nn (k=5):",round(accuracy_score(y_test, y_pred5),3))


#knn with neighbor=10
knn10 = neighbors.KNeighborsClassifier(n_neighbors=10)
knn10.fit(X_train, y_train)

y_pred10=knn10.predict(X_test)

print("Accuracy of k-nn (k=10):",round(accuracy_score(y_test, y_pred10),3))

