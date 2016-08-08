
from __future__ import division
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import random
import csv
import math


'''Random Forest Machine Learning Classifier

   Takes in features.csv as input data and assigns binary classification labels using a random forest of low depth
   decision trees as guesstimate.csv.
   Random sampling of decision trees to find classification model leads to slight variance in output with each run
   iteration.
   Uses sklearn.py as base for decision tree implementation'''

class RandomForest:
    def __init__(self, num_trees):
        self.num_trees = num_trees  # number of decision trees in random forest
        self.trees = []             # array of trees
        self.tree_features = []     # array of features in each tree
        self.tree_weights = []      # array of weights
        self.num_features = 10      # number of features per tree. approx sqrt(37). [37 = number of features in forest]

    '''Takes args and generates model using helper method generate_trees'''
    def fit(self, data, features, labels):
        self.generate_trees(data, features, labels)
        self.tree_weights = [1]*self.num_trees

    '''Creates decision trees for random forest'''
    def generate_trees(self, data, features, labels):

        # loop to generate trees
        for i in range(0, self.num_trees):

            # create a decision tree
            t = DecisionTreeClassifier()

            # create list of features for the decision tree using random sample
            feat_index = random.sample(range(0,50), self.num_features)
            feat = []
            for index in feat_index:
                feat.append(features[index])
            feat = pd.Index(feat)
            self.tree_features.append(feat)

            num_samples = np.random.randint(int(len(labels)/4), len(labels))
            indexes = np.random.choice(data.index, num_samples, replace=True)
            samples = data.loc[indexes]

            # fit randomly sampled features to decision tree model
            t.fit(samples[feat], samples['outcome'])
            self.trees.append(t)

    '''Assign labels to data points'''
    def predict_proba(self, data):

        frame = None # 2-D array that holds probabilistic label predictions for each data point
        first = True # boolean flag for first tree in forest

        # loop to calculate probability and store in frame
        for i in range(0, len(self.trees)):

            # get probability for first decision tree
            if first:
                first = False
                frame = self.trees[i].predict_proba(data[self.tree_features[i]])

            # get probability for subsequent decision trees
            else:
                res = self.trees[i].predict_proba(data[self.tree_features[i]])  # get probability for decision tree
                for i in range(0, len(frame)):
                    for j in range(0, len(frame[0])):
                        frame[i][j] = frame[i][j] + res[i][j]  # update probability

        # loop over each point to convert probability to be between 0 and 1
        for i in range(0, len(frame)):
            for j in range(0, len(frame[0])):
                frame[i][j] = frame[i][j]/len(self.trees)

        return frame

    '''Calculate score out of 100 for classifier'''
    def score(self, data, labels):
        labels = labels.tolist()
        guess = self.predict_proba(data)
        guess = np.transpose(guess)[1]
        error = 0

        # loop over each point and calculate error
        for i in range(0, len(labels)):
            error += math.fabs(labels[i] - guess[i])

        # print score out of 100
        print(100.0 - (error/len(labels))*100)


'''Set up method called by main to convert input data into workable form'''
def work():

    # convert input csv to pandas data frame
    data = pd.read_csv('features.csv')

    # fix missing values, data type mismatches, etc
    data = data.interpolate()

    # define label for bot as outcome = 1
    is_bot = data['outcome'] == 1

    # separate training data from data
    train, final = data[data['outcome'] >= 0], data[data['outcome'] == -1]

    # randomly sample 75% of training data as learning data for model. reserve remaining 25% as validation data
    train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
    learn, test = train[train['is_train'] == True], train[train['is_train'] == False]

    # select features from data
    features = data.columns[56:58] | data.columns[75:76] | data.columns[77:92] | data.columns[164:186] | data.columns[196:197] | data.columns[198:202] | data.columns[403:412] | data.columns[204:206] | data.columns[207:209] | data.columns[210:214]

    y, _ = pd.factorize(train['outcome'])

    # generate random forest of 5000 trees, fit model, then score model
    rf = RandomForest(5000)
    rf.fit(train, features, y)
    rf.score(test, test['outcome'])

    # generate labels for test data and output as csv file
    probs = rf.predict_proba(final)
    res = {}
    res['prediction'] = np.transpose(probs)[1]
    res = pd.DataFrame(res, index=range(2013, 6713))
    res = final[final.columns[1:2]].join(res)
    res.to_csv('guesstimate9.csv', encoding='utf-8', index=False)

if __name__ == '__main__':
    work()
