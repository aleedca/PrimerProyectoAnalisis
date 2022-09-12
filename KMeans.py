import random

class KMEANS:

    #constructor
    def __init__(self, k):
        self.MAXVECTOR = 5 #max could be 10,000

        self.vector = []
        self.centroids = []

        self.K = k #must be greater than 0
        #hacer diccionario con labels para clasificarlos

    #methods
    def generateVector(self):
        for index in range(self.MAXVECTOR):
            self.vector.append([random.randint(0,100),random.randint(0,100)])

    def createRandomCentroids(self):
        self.centroids.clear()
        for index in range(self.K):
            randomNumber = random.randint(0,self.MAXVECTOR-1)
            self.centroids.append(self.vector[randomNumber])

    def kmeans(self):
        self.createRandomCentroids()
        for element in self.centroids:
            print("Centroid:", element)

        print("distancias")

    def printVector(self):
        for element in self.vector:
            print(element, end = "")
        print()

    #main
    def main(self):
        self.generateVector()
        self.printVector()
        self.kmeans()

if __name__ == "__main__":
    kmeans = KMEANS(3)
    kmeans.main()