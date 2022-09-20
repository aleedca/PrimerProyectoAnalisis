#Loading the required modules
import time
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist 
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn import datasets

executedLines = 0 
comparations = 0
asignations = 0 

#Defining our function 
def kmeans(x, k, no_of_iterations):
    idx = np.random.choice(x.shape[0], k, replace=False)

    #Randomly choosing Centroids 
    centroids = x.iloc[idx, :] #Step 1
     
    #finding the distance between centroids and all the data points
    distances = cdist(x, centroids ,'euclidean') #Step 2
     
    #Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances]) #Step 3

    #Repeating the above steps for a defined number of iterations
    #Step 4
    for _ in range(no_of_iterations): 
        centroids = []
        for idx in range(k):
            #Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points==idx].mean(axis=0) 
            centroids.append(temp_cent)
 
        centroids = np.vstack(centroids) #Updated Centroids 
         
        distances = cdist(x, centroids ,'euclidean')
        points = np.array([np.argmin(i) for i in distances])
         
    return points 

#load data
surveys_df = pd.read_csv("data/surveys.csv")
df = pd.DataFrame(np.random.randint(0,100,size=(500, 3)), columns=list('ABC'))

#execution
start = time.time()
results = kmeans(df,3,100)
end = time.time()
executionTime = end - start

print(executionTime)
#print(results)