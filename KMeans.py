#Loading the required modules
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from scipy.spatial.distance import cdist 
from sklearn import datasets
from sklearn.cluster import KMeans

 
#Defining our function 
def kmeans(x, k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    #Randomly choosing Centroids 
    
    #x = x[1]
    #print("lol", type(idx))
    centroids = x[idx, :] #Step 1
    
     
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

surveys_df = pd.read_csv("surveys.csv")
print(surveys_df[1])
kmeans(surveys_df["year"],3,1000)
