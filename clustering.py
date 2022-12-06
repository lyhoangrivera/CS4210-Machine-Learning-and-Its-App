#-------------------------------------------------------------------------
# AUTHOR: Ly Hoang
# FILENAME: clustering
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df.values)

bestK = 0
bestSilhouette = 0
scores = []
#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
for k in range(2, 20):
     kmeans = KMeans(n_clusters=k, random_state=0)
     kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
     score = silhouette_score(X_training, kmeans.labels_)
     scores.append(score)

     if score > bestSilhouette:
          bestSilhouette = score
          bestK = k
#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
plt.scatter([i for i in range(2, 20)], scores, alpha=0.5)
plt.xlabel('k')
plt.ylabel('silhouette coefficient')
plt.title('K vs Silhouette coefficient')
plt.show()

#reading the test data (clusters) by using Pandas library
#--> add your Python code here
df = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
kmeans = KMeans(n_clusters=bestK, random_state=0)
kmeans.fit(X_training)

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(np.array(df.values).reshape(1, len(df.values))[0], kmeans.labels_).__str__())
#--> add your Python code here
