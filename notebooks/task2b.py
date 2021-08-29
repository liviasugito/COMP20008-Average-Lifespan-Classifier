import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
import csv
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mutual_info_score
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

#load in the data
world = pd.read_csv('world.csv',encoding = 'ISO-8859-1')
life = pd.read_csv('life.csv',encoding = 'ISO-8859-1')

#combine to get feature and label
world_life_join = pd.merge(world, life, how='right', on='Country Code')

#get just the features
world_life_join = world_life_join.replace('..', np.nan)
data = world_life_join.iloc[:,3:-3].astype(float)
feature_label = data.columns
#print(feature_label)
#get just the class labels
classlabel = world_life_join.iloc[:,-1]

##randomly select 2/3 of the instances to be training and the rest to be testing
X_train, X_test, y_train, y_test = train_test_split(data,classlabel, train_size=2/3, test_size=1/3, random_state=100)

#impute missing value using median of each column
imp_med = SimpleImputer(missing_values=np.nan, strategy='median')
imp_med = imp_med.fit(X_train)
X_train = imp_med.transform(X_train)
X_test = imp_med.transform(X_test)

#Multiply columns in X_train
X_train_df = pd.DataFrame(X_train)
X_train_df.columns = feature_label
l = list(combinations(X_train_df.columns,2))
df_add = pd.concat([X_train_df[col[1]] * (X_train_df[col[0]]) for col in l], axis=1, keys=l)
df_add.columns = df_add.columns.map(''.join)
joined_train_data = X_train_df.join(df_add)
joined_train_data.to_csv('task2bcsv1.csv')
print("Number of row and column of training features after multiplication:",joined_train_data.shape)

#use scaled data for K-Mean
X_train_scaled = X_train
X_test_scaled = X_test
scaler = preprocessing.StandardScaler().fit(X_train_scaled)
X_train_scaled = scaler.transform(X_train_scaled)
X_test_scaled = scaler.transform(X_test_scaled)

K = range(1,7)
sum_of_squared_distance = []

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=100)
    kmeans = kmeans.fit(X_train_scaled)
    sum_of_squared_distance.append(kmeans.inertia_)
f = plt.figure()
plt.plot(K, sum_of_squared_distance, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distance')
plt.title('Elbow Method for Optimal K')
f.savefig('task2bgraph1.png', bbox_inches = 'tight')

#transform High, Low, Medium to 0, 1, 2
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


#use 3 as number of clusters
kmeans = KMeans(n_clusters=3, random_state=100).fit(X_train_scaled)

centroids = kmeans.cluster_centers_
#df = pd.crosstab(y_train, kmeans.labels_)
#df



#graph to compare the clustering to y_train - actual training label
color_theme = np.array(['darkgray','lightsalmon','powderblue'])
relabel = np.choose(kmeans.labels_,[0,1,2])
fig = plt.figure()
plt.subplot(1,2,1)
plt.scatter(x=X_train_df.iloc[:,1],y=X_train_df.iloc[:,0],c=color_theme[y_train])
plt.xlabel("Adjusted Savings")
plt.ylabel("Access to electricity")
plt.title("Actual Result from Label (y_train)")



plt.subplot(1,2,2)
plt.scatter(x=X_train_df.iloc[:,1],y=X_train_df.iloc[:,0],c=color_theme[relabel])
plt.xlabel("Adjusted Savings")
plt.title('Kmeans Clustering Result')
fig.savefig('task2bgraph2.png')
#print(classification_report(y_train, relabel, target_names=["High","Low","Medium"]))



joined_train_data['cluster_label'] = kmeans.labels_
print("Number of row and column of training features after adding feature from clustering:",joined_train_data.shape)
joined_train_data.to_csv('task2bcsv2.csv')
#Multiply columns in X_test
X_test_df = pd.DataFrame(X_test)
X_test_df.columns = feature_label


l_test = list(combinations(X_test_df.columns,2))
df_add_test = pd.concat([X_test_df[col[1]] * (X_test_df[col[0]]) for col in l_test], axis=1, keys=l)
df_add_test.columns = df_add_test.columns.map(''.join)
joined_test_data = X_test_df.join(df_add_test)
print("Number of row and column of test features after multiplication:",joined_test_data.shape)
joined_test_data.to_csv('task2bcsv3.csv')

#Assign each point in X_test to nearest centroids obtained from K-Mean on training set
test_labels = kmeans.predict(X_test_scaled)
joined_test_data['cluster_label'] = test_labels
print("Number of row and column of test features after adding feature from clustering:",joined_test_data.shape)
joined_test_data.to_csv('task2bcsv4.csv')
selector = SelectKBest(mutual_info_classif, k=4)
best4_data = selector.fit_transform(joined_train_data, y_train)
cols = selector.get_support(indices=True)
#print(cols)
best4_test = joined_test_data.iloc[:,cols]

#chosen cols
joined_train_data.iloc[:,cols]

#scale the 4 chosen best features
scaler = preprocessing.StandardScaler().fit(best4_data)
best4_data=scaler.transform(best4_data)
best4_test=scaler.transform(best4_test)

knn_fe = neighbors.KNeighborsClassifier(n_neighbors=5)
knn_fe.fit(best4_data, y_train)

y_pred_fe=knn_fe.predict(best4_test)

print("Accuracy of feature engineering:"+str(round(accuracy_score(y_test, y_pred_fe),3)))


X_train2 = X_train_scaled
X_test2 = X_test_scaled
y_train2 = y_train
y_test2 = y_test


#fit ONLY to training set
pca = PCA(n_components = 4)
pca = pca.fit(X_train2)

#apply mapping to both training set and test set
X_train_pca = pca.transform(X_train2)
X_test_pca = pca.transform(X_test2)


#use 5nn to train the model
knn_pca = neighbors.KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train2)

y_pred_pca=knn_pca.predict(X_test_pca)

print("Accuracy of PCA: "+str(round(accuracy_score(y_test2, y_pred_pca),3)))


#using first four features
X_train_4f = X_train_scaled[:,0:4]
knn5 = neighbors.KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train_4f, y_train)

y_pred5=knn5.predict(X_test_scaled[:,0:4])

print("Accuracy of first four features: "+str(round(accuracy_score(y_test, y_pred5),3)))
