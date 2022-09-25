#Loading the required modules
from scipy.spatial.distance import cdist 
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
    assignments += 4

    #Randomly choosing Centroids 
    centroids = x.iloc[idx, :] #Step 1
    assignments += 1
     
    #Finding the distance between centroids and all the data points
    distances = cdist(x, centroids ,'euclidean') #Step 2
    assignments += 4
     
    #Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances]) #Step 3
    assignments += len(distances) + 1

    #Repeating the above steps for a defined number of iterations
    for _ in range(no_of_iterations): #Step 4
        centroids = []
        assignments += 2
        comparisons += 1
        for idx in range(k):
            #Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points==idx].mean(axis=0) 
            centroids.append(temp_cent)
            assignments += 4
            comparisons += 2
            executedLines += 2
 
        centroids = np.vstack(centroids) #Updated Centroids 
        assignments += 2
         
        distances = cdist(x, centroids ,'euclidean')
        assignments += 4

        points = np.array([np.argmin(i) for i in distances])
        assignments += len(distances) + 1
        
        comparisons += 1
        executedLines += 5
         
    comparisons += 1
    executedLines += 6
    return points 

def main():
    #Load data
    n = 100
    df = pd.DataFrame(np.random.randint(0,100,size=(100, 3)), columns=list('ABC'))

    #Execution
    start = time.time()
    results = kmeans(df,3,n) #N iterations
    end = time.time()
 
    print("Execution Time", (end - start))
    print("Executed Lines:", executedLines)
    print("Comparisons:", comparisons)
    print("Assignments:", assignments)
    #print(results)

main()