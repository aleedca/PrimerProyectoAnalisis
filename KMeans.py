#Loading the required modules
from scipy.spatial.distance import cdist 
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from datetime import timedelta
from sklearn import datasets

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import time

#Setting variables
assignments = 0
comparisons = 0
executedLines = 0

#Defining our function 
def kmeans(x, k, no_of_iterations):
    global assignments
    global comparisons
    global executedLines
   
    idx = np.random.choice(x.shape[0], k, replace=False)
    assignments += 3

    #Randomly choosing Centroids 
    centroids = x.iloc[idx, :] #Step 1
    assignments += 1
     
    #finding the distance between centroids and all the data points
    distances = cdist(x, centroids ,'euclidean') #Step 2
    assignments += 3
     
    #Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances]) #Step 3
    assignments += len(distances)

    #Repeating the above steps for a defined number of iterations
    for _ in range(no_of_iterations): #Step 4
        centroids = []
        assignments += 2
        comparisons += 1
        for idx in range(k):
            #Updating Centroids by taking mean of Cluster it belongs to
            assignments += 4
            comparisons += 2
            temp_cent = x[points==idx].mean(axis=0) 
            centroids.append(temp_cent)
            executedLines += 2
 
        centroids = np.vstack(centroids) #Updated Centroids 
        assignments += 1
         
        distances = cdist(x, centroids ,'euclidean')
        assignments += 3

        points = np.array([np.argmin(i) for i in distances])
        assignments += len(distances)
        
        comparisons += 1
        executedLines += 5
         
    comparisons += 1
    executedLines += 6
    return points 

def main():
    #load data
    n = 100 
    surveys_df = pd.read_csv("data/surveys.csv")
    df = pd.DataFrame(np.random.randint(0,100,size=(n, 3)), columns=list('ABC'))

    #execution
    start = time.time()
    results = kmeans(df,3,100)
    end = time.time()

    print("Execution Time:", str(timedelta(seconds = end - start)))
    print("Executed Lines:", executedLines)
    print("Assignments:", assignments)
    print("Comparisons:", comparisons)
    #print(results)

main()