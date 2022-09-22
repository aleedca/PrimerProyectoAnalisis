import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split


assignments = 0
comparisons = 0
executedLines = 0

class Node:
    '''
    Helper class which implements a single tree node.
    '''

    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value

class DecisionTree:
    '''
    Class which implements a decision tree classifier algorithm.
    '''

    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    @staticmethod
    def _entropy(s):
        global assignments
        global comparisons
        global executedLines
        '''
        Helper function, calculates entropy from an array of integer values.

        :param s: list
        :return: float, entropy value
        '''
        # Convert to integers to avoid runtime errors
        counts = np.bincount(np.array(s, dtype=np.int64))
        assignments += 2
        # Probabilities of each class label
        percentages = counts / len(s)
        assignments += 1

        # Caclulate entropy
        entropy = 0
        assignments += 1
        for pct in percentages:
            assignments += 1
            comparisons += 1
            executedLines += 1
            if pct > 0:
                entropy += pct * np.log2(pct)
                assignments += 2
                executedLines += 1

        executedLines += 5
        return -entropy

    def _information_gain(self, parent, left_child, right_child):
        global assignments
        global comparisons
        global executedLines
        '''
        Helper function, calculates information gain from a parent and two child nodes.

        :param parent: list, the parent node
        :param left_child: list, left child of a parent
        :param right_child: list, right child of a parent
        :return: float, information gain
        '''
        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)
        assignments += 2

        # One-liner which implements the previously discussed formula
        assignments += 3
        executedLines += 3
        return self._entropy(parent) - (num_left * self._entropy(left_child) + num_right * self._entropy(right_child))

    def _best_split(self, X, y):
        global assignments
        global comparisons
        global executedLines
        '''
        Helper function, calculates the best split for given features and target

        :param X: np.array, features
        :param y: np.array or list, target
        :return: dict
        '''
        best_split = {}
        best_info_gain = -1
        n_rows, n_cols = X.shape
        assignments += 4

        # For every dataset feature
        for f_idx in range(n_cols):
            assignments += 2
            comparisons += 1
            X_curr = X[:, f_idx]
            executedLines += 1
            # For every unique value of that feature
            for threshold in np.unique(X_curr):
                assignments += 1
                comparisons += 1
                # Construct a dataset and split it to the left and right parts
                # Left part includes records lower or equal to the threshold
                # Right part includes records higher than the threshold
                #print("----------------------------------------",X_curr)
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                assignments += 6
                df_left = np.array([row for row in df if row[f_idx] <= threshold])
                df_right = np.array([row for row in df if row[f_idx] > threshold])

                assignments += 2 + (2 * (len(df)))
                comparisons += 2 * len(df)

                # Do the calculation only if there's data in both subsets
                comparisons += 2
                executedLines += 4
                if len(df_left) > 0 and len(df_right) > 0:
                    # Obtain the value of the target variable for subsets
                    y = df[:, -1]
                    y_left = df_left[:, -1]
                    y_right = df_right[:, -1]
                    assignments += 4
                    # Caclulate the information gain and save the split parameters
                    # if the current split if better then the previous best
                    gain = self._information_gain(y, y_left, y_right)
                    comparisons += 1
                    executedLines += 5
                    if gain > best_info_gain:
                        best_split = {
                            'feature_index': f_idx,
                            'threshold': threshold,
                            'df_left': df_left,
                            'df_right': df_right,
                            'gain': gain
                        }
                        best_info_gain = gain
                        executedLines += 2
                        assignments += 2
        comparisons += 1 

        executedLines += 5   
        return best_split

    def _build(self, X, y, depth=0):
        global assignments
        global comparisons
        global executedLines
        '''
        Helper recursive function, used to build a decision tree from the input data.

        :param X: np.array, features
        :param y: np.array or list, target
        :param depth: current depth of a tree, used as a stopping criteria
        :return: Node
        '''
        n_rows, n_cols = X.shape

        assignments += 2
        comparisons += 2
        # Check to see if a node should be leaf node
        if n_rows >= self.min_samples_split and depth <= self.max_depth:
            # Get the best split
            best = self._best_split(X, y)
            assignments += 1
            comparisons += 1
            executedLines += 2
            # If the split isn't pure
            if best['gain'] > 0:
                # Build a tree on the left
                left = self._build(
                    X=best['df_left'][:, :-1],
                    y=best['df_left'][:, -1],
                    depth=depth + 1
                )
                assignments += 4

                right = self._build(
                    X=best['df_right'][:, :-1],
                    y=best['df_right'][:, -1],
                    depth=depth + 1
                )
                assignments += 9
                executedLines += 3
                return Node(
                    feature=best['feature_index'],
                    threshold=best['threshold'],
                    data_left=left,
                    data_right=right,
                    gain=best['gain']
                )
        # Leaf node - value is the most common target value
        assignments += 1
        executedLines += 3
        return Node(
            value=Counter(y).most_common(1)[0][0]
        )

    def fit(self, X, y):
        global assignments
        global comparisons
        global executedLines
        '''
        Function used to train a decision tree classifier model.

        :param X: np.array, features
        :param y: np.array or list, target
        :return: None
        '''
        # Call a recursive function to build the tree
        self.root = self._build(X, y)
        executedLines += 1
        assignments += 1

    def _predict(self, x, tree):
        global assignments
        global comparisons
        global executedLines
        '''
        Helper recursive function, used to predict a single instance (tree traversal).

        :param x: single observation
        :param tree: built tree
        :return: float, predicted class
        '''
        # Leaf node
        comparisons += 1
        if tree.value != None:
            executedLines += 1
            return tree.value
        feature_value = x[tree.feature]
        assignments += 1

        # Go to the left
        comparisons += 1
        if feature_value <= tree.threshold:
            assignments += 2
            executedLines += 1
            return self._predict(x=x, tree=tree.data_left)

        # Go to the right
        comparisons += 1
        executedLines += 4
        if feature_value > tree.threshold:
            assignments += 2
            executedLines += 1
            return self._predict(x=x, tree=tree.data_right)

    def predict(self, X):
        global assignments
        global comparisons
        global executedLines
        '''
        Function used to classify new instances.

        :param X: np.array, features
        :return: np.array, predicted classes
        '''
        # Call the _predict() function for every observation
        assignments += 2 + len(X)
        executedLines += 1
        return [self._predict(x, self.root) for x in X]

def main():
    num = 7000

    df = pd.DataFrame(np.random.randint(0,100,size=(num, 4)), columns=list('ABCD'))
    df['label'] = np.random.randint(1,4,size=(num,1))

    X = df[['A','B','C']].values
    y = df[['label']].values
   
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    start = time.time()
    model = DecisionTree()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    end = time.time()
    print(preds)
    print("========================================================================================================")
    print("Execution Time",(end-start))
    print("Executed Lines:", executedLines)
    print("Assignments:", assignments)
    print("Comparisons:", comparisons)


main()