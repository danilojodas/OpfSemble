# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:04:58 2020

@author: DANILO

THIS CODE WAS ADAPTED FROM THE EXAMPLE AVAILABLE AT https://machinelearningmastery.com/super-learner-ensemble-in-python/
AND CREATED BY Jason Brownlee. WE INCLUDE A WIDE RANGE OF CHANGES TO PERFORM THE INSTRUCTIONS FOR ENSEMBLE PRUNING BASED ON THE OPF.
"""

from numpy import vstack
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import DBSCAN
from scipy.stats import mode
from ensemble import Ensemble
from sklearn.metrics import accuracy_score,f1_score
from divergence_measures import disagreement_matrix,paired_q_matrix,kl_divergence_matrix
from collections import defaultdict
import math
import numpy as np
import sys

import logging
logging.disable(sys.maxsize)

class DBScanSemble:
    """
    A class which implements the DBScan Ensemble Learning.
    """

    def __init__(self, n_models=10, n_folds=10,n_classes=0,ensemble=None,meta_data_mode='count_class',divergence=None,bootstrapping=False,random_state=None):
        """
        Initialization of the class properties.

        Parameters
        ----------
        n_models: int
            The number of models to be created. Default is 10
        n_folds: int
            The number of folds of the cross validation. Default is 10
        n_classes: int
            The number of classes in the dataset. It is usually automatically determined in the fitting of the model. Default is 0        
        ensemble: list
            A list with the baseline classifiers of the ensemble model. If not specified, a random list will be created according to n_models. Default is None
        meta_data_model: str
            The approach to create the meta-data from the cross validation predictions. Default is 'count_class'
        divergence: bool
            It specifies whether to apply or not the Kullback-Lieber divergence on the meta-data. Default is False
        bootstrapping: bool
            Indicates whether to apply the boostrapping during the training of the estimators. Default is False
        random_state: int,None
            Seed value for the bootstrapping approach. Default is None
        """

        # Check for a valid meta_data_mode
        if (not meta_data_mode in ['oracle','count_class']):
            raise SystemExit('Value for the meta_data_mode is not valid. Please inform one of the following mode: ',['oracle','count_class'])

        # Check for a valid divergence metric
        if (not divergence in [None,'yule','disagreement','kullback-lieber']):
            raise SystemExit('Divergence metric must be one of the following: yule, disagreement,kullback-lieber')            

        if (ensemble != None):
            if (type(ensemble) == Ensemble):
                print('Build ensemble from an Ensemble object...')
                self.ensemble = ensemble
            elif(type(ensemble) == list):
                print('Build ensemble from a list of pre-defined classifiers...')                 
                self.ensemble = Ensemble(n_models=len(ensemble),models=ensemble)
        else:        
            self.ensemble = Ensemble(n_models=n_models)
        
        self.n_models = len(self.ensemble.items)
        self.n_folds = n_folds
        self.centroids = None
        self.clusters = None
        self.centroids_scores = None
        self.n_classes = n_classes

        self.meta_data_mode = meta_data_mode
        self.divergence = divergence

        self.bootstrapping = bootstrapping
        self.random_state = random_state
        self.max_features = None
        self.max_samples = None

    # create a list of base-models
    def get_models(self):
        """
        Creates and returns a list with the default base-models.

        Returns
        -------
        models: list
            A list with the base-models
        """

        models = list()
        models.append(LogisticRegression(max_iter=22))
        models.append(KNeighborsClassifier(n_neighbors=1))
        models.append(GradientBoostingClassifier(learning_rate=0.09, n_estimators=500))
        models.append(RandomForestClassifier(n_estimators=10, criterion='entropy'))
        models.append(SVC(C=61, gamma=0.08, kernel='rbf', degree=1, probability=True))

        return models

    def fit(self, X, y):
        """
        Training of the base-models and the meta-model

        Parameters
        ----------
        X: array
            A 2D array with the training set
        y: array
            A column array with the labels of each samples in X        
        """

        # Check for a valid meta_data_mode
        if (not self.meta_data_mode in ['oracle','count_class']):
            raise SystemExit('Value for the meta_data_mode is not valid. Please inform one of the following mode: ',['oracle','count_class'])        
        
        meta_X = np.array([], dtype=int)
        if self.n_classes==0:
            self.n_classes = len(np.unique(y))
            if self.n_classes<np.max(y):
                self.n_classes = np.max(y)

        # define split of data
        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        # enumerate splits
        for train_ix, test_ix in kfold.split(X):
            fold_res = list()
            # get data
            train_X, test_X = X[train_ix], X[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            
            # fit and make predictions with each sub-model
            for i,item in enumerate(self.ensemble.items):
                model = item.classifier
                
                # Checks if bootstrapping is enabled
                if (self.bootstrapping):
                    X_boot,y_boot = self.__bootstrap_sample(train_X,train_y,i)
                    model.fit(X_boot,y_boot)
                else:
                    model.fit(train_X, train_y)

                yhat = model.predict(test_X)
            
                item.score += f1_score(yhat, test_y, average='weighted')/self.n_folds
                if (self.meta_data_mode=='oracle'):
                    res = np.copy((yhat == test_y).astype(int))
                else:
                    res = np.copy(yhat)

                # store columns
                fold_res.append(res)
        
            # store fold yhats as columns
            meta_X = np.hstack([meta_X, vstack(fold_res)]) if meta_X.size else vstack(fold_res)

        if (self.meta_data_mode=='count_class'):
            # Creating a new vector with the baseline predictions' counts for each class
            new_x = np.zeros((self.n_models,self.n_classes))
            
            for i,_ in enumerate(meta_X):
                un,counts = np.unique(meta_X[i],return_counts=True)
                for j,c in enumerate(un):
                    # Minimum label value must be 0 for the new_x indexing
                    c = c if np.min(un) == 0 and c < self.n_classes else c - 1
                    new_x[i,c] = counts[j]
        else:
            new_x = np.copy(meta_X)
        
        # Apply divergence (or not) to meta_X (only if meta_data_mode is oracle)
        if (not self.divergence is None and self.meta_data_mode=='oracle'):
            if self.divergence=='yule':
                new_x = paired_q_matrix(new_x.T)
            elif self.divergence=='disagreement':
                new_x = disagreement_matrix(new_x.T)
            elif self.divergence=='kullback-lieber':
                new_x = kl_divergence_matrix(new_x)
            
        return new_x

    def predict(self, X, voting='mode'):
        """
        Perform the classification of the data in X with stacked model

        Parameters
        ----------
        X: array
            A 2D array with the testing/validation set
        voting: array
            A string representing the voting approach ('intracluster','mode','average','intercluster','mode_best' or 'aggregation')

        Returns
        -------
        predictions: array
            A column array with the classification results of each sample in X
        """

        if self.centroids is None:
            raise SystemExit('Meta model was not fitted!')
            
        if self.clusters is None:
            raise SystemExit('No cluster is defined! It might be happened because the model is not fitted yet!')

        voting_options = ['intracluster','mode','average','intercluster','mode_best','aggregation']

        if (not voting in voting_options):
            raise SystemExit('The informed voting for predict is not compatible with a valid option. Please, inform one of the following options: ',voting_options)

        if voting=='intracluster':
            # Predictions from the classifiers belonging to the protopype with the highest F1-score
            
            # Getting the key of the best-performing prototype, i.e., the prototype with the highest F1-score
            max_key = max(self.centroids_scores, key=self.centroids_scores.get)
            
            # Getting the ID of the best-performing prototype assigned by the Unsupervised OPF
            idx_prot = np.where([i.key == max_key for i in self.ensemble.items])[0][0]
            
            # Getting the classifiers from the best-performing cluster
            clfs = self.clusters[idx_prot]
            
            # Getting predictions from classifiers
            preds = []
            for c in clfs:
                preds.append(c.predict(X))
            
            # Getting the final predictions based on the most common predicted class among the classifiers
            pred = mode(np.asarray(preds),axis=0)[0].reshape(-1,1)  
        else:
            preds = []
            scores = []
    
            for key in self.centroids:
                model = self.centroids[key]
                scores.append(self.centroids_scores[key])
                pr = model.predict(X)
                preds.append(pr)
    
            scores = np.asarray(scores)
            preds = np.asarray(preds)
    
            if voting=='mode':
                #major voting
                pred, _ = mode(preds, axis=0)
            elif voting=='average':
                #compute the prototypes average score by label
                mat = np.zeros((self.n_classes, len(self.centroids)))
                pred = np.zeros(len(X))
                for sample in range(len(X)):
                    for prot in range(len(self.centroids)):
                        mat[preds[prot,sample]-1,prot] = scores[prot]
                    
                    summing = np.zeros(len(self.centroids))
                    count = np.zeros(len(self.centroids))
                    
                    for prot in range(len(self.centroids)):
                        if mat[preds[prot,sample]-1,prot]>0:
                            summing[prot]+=scores[prot]
                            count[prot]+=1
                    pred_avgs = np.zeros(len(self.centroids))
                    
                    for prot in range(len(self.centroids)):
                        pred_avgs[prot] = summing[prot]/count[prot]
    
                    pred[sample] = preds[np.argmax(pred_avgs),sample]
            elif voting=='intercluster':
                preds_each_cluster = list()
                
                for key in self.clusters: # iterate in each cluster to obtain its classifiers
                    preds = list()
                    
                    for model in self.clusters[key]: # iterate in each model of the current cluster                  
                        preds.append(model.predict(X))
                    
                    preds = np.asarray(preds)
                    
                    # Mode for the current cluster (intracluster voting)
                    preds_each_cluster.append(mode(preds, axis=0)[0])
                            
                preds_each_cluster = np.vstack(preds_each_cluster)
                
                pred = preds_each_cluster        
                
                # Mode for all clusters (intercluster voting)
                # Minimum of two clusters to calculate the mode
                if (len(preds_each_cluster) > 1):
                    pred = mode(preds_each_cluster, axis=0)[0]
                else:
                    pred = preds_each_cluster
                
            elif voting=='mode_best':
                max_cluster_value = np.zeros(len(self.clusters))
                max_cluster_index = np.zeros(len(self.clusters)).astype(int)
                for i in range(len(self.ensemble.items)):
                    item = self.ensemble.items[i]
                    for j in range(len(self.clusters)):
                        if(item.cluster_id==j): 
                            if max_cluster_value[j]<item.score:
                                max_cluster_value[j]=item.score
                                max_cluster_index[j]=i
                preds_best_cluster = []
                for j in range(len(self.clusters)):
                    model = self.ensemble.items[max_cluster_index[j]].classifier
                    pr = model.predict(X)
                    preds_best_cluster.append(pr)
                preds_best_cluster= np.asarray(preds_best_cluster)
                pred, _ = mode(preds_best_cluster, axis=0)
            elif voting=='aggregation': # aggregation of all voting methods
                voting_methods = [v for v in voting_options if v != 'aggregation']
                preds = list()

                # Get predictions from all the voting methods
                for v in voting_methods:
                    preds.append(self.predict(X,voting=v).reshape(-1,1))

                #print(np.asarray(preds))
                pred = mode(np.asarray(preds),axis=0)[0][0]

        return pred.reshape(-1,1)

    # fit a meta model
    def fit_meta_model(self, X, radius=3):
        """
        Perform the DBSCAN clustering over the classifiers attributes and returns the prototype classifiers

        Parameters
        ----------
        X: array
            A 2D array with the classifiers attributes
        radius: float
            The value of radius to be used in the DBSCAN algorithm
        """

        try:
            dbscan = DBSCAN(eps=radius)
            dbscan.fit(X)

            labels = dbscan.labels_
            unique_labels = np.unique(labels)

            n_clusters = len(unique_labels)
            print(f'Unique labels: {unique_labels}')
            print(f'Labels: {labels}')

            centroids_idx = []
            clusters = defaultdict(list)

            # Iterate over each cluster
            for i, cluster_id in enumerate(unique_labels):
                # Check if the cluster is noise
                if cluster_id == -1:
                    continue

                # Getting the centroids of each cluster
                cluster_indices = np.where(labels == cluster_id)[0]
                if len(cluster_indices) == 0:
                    continue

                cluster_points = X[cluster_indices]

                centroid = np.mean(cluster_points, axis=0)

                # Find the index of the classifier closest to the centroid
                distances = np.linalg.norm(X[cluster_indices] - centroid, axis=1)
                closest_index = cluster_indices[np.argmin(distances)]
                centroids_idx.append(closest_index)

                for j in range(len(cluster_indices)):
                    self.ensemble.items[cluster_indices[j]].cluster_id = closest_index
                    # Adding models to the corresponding centroid in order to form a dictionary of clusters
                    clusters[closest_index].append(self.ensemble.items[cluster_indices[j]].classifier)

            centroids = dict()
            centroids_scores = dict()   
            # Getting the centroid classifiers and their associated F1-scores     
            for i in range(self.ensemble.n_models):
                for j in range(len(centroids_idx)):
                    if(i==centroids_idx[j]):      
                        item=self.ensemble.items[i]
                        key = item.key
                        centroids[key] = item.classifier
                        centroids_scores[key] = item.score           

            self.centroids =  centroids
            self.centroids_scores =  centroids_scores
            self.clusters = clusters
        except Exception as e:
            print(f'Something went wrong while fitting the meta-model with {n_clusters} clusters! Error: {e}')

    def __bootstrap_sample(self,X,y,counter=0):
        """
        Generate a bootstrap sample with replacement from a 2D array.

        Args:
            X (array): Input samples.
            y (array): Labels of the input samples.
            counter (int): Value to increment the seed of the random state.

        Returns:
            bootstrap_sample (array): Bootstrap sample of the same shape as the input data.
        """



        if (not self.random_state is None):
            np.random.seed(self.random_state+counter)

        n_samples,n_features = X.shape
        
        # Bootstrapping for the samples
        if (self.max_samples is None):
            m = n_samples
        else:
            if (self.max_samples > n_samples):
                m=n_samples
            else:
                m=self.max_samples
        
        # Bootstrapping for the features
        if (self.max_features == 'sqrt'):
            n = round(sqrt(n_features))
        elif (type(self.max_features) == int):
            if (self.max_features > n_features):
                n = n_features
            else:
                n = self.max_features
        else:
            n = n_features
        
        # Bootstrapping for the samples
        idx_samples = np.random.choice(n_samples, m, replace=True)
        # Bootstrapping for the features
        idx_feats = np.sort(np.random.choice(n_features, n, replace=False))
        # Selects the bootstrap samples and their associated bootstrap features
        X_sub = X[idx_samples, :]
        X_sub = X_sub[:,idx_feats]
        y_sub = y[idx_samples]
        
        return X_sub,y_sub        

    def get_scores_baselines(self,X,y,path=None):
        '''
            Function that computes the accuracy and the F1-score from each baseline regression model that compound the ensemble.

            Parameters
            ----------
            X : array
                Array with the samples to be predicted
            y : array
                Array with output values of each test sample
            path: str
                Path to the folder where the baseline results will be saved. Default is None

            Returns
            -------
            scores : dictionary
                A dictionary where each key is the baseline model's name and the value is a list with the MAE and MSE computed from the model
        '''

        scores = {}
        average_f1 = 'binary'

        # Assigning the F1 average based on the number of classes
        n_classes = len(np.unique(y))

        if (n_classes > 2):
            average_f1 = 'macro'

        for item in self.ensemble.items:
            model = item.classifier
            y_pred = model.predict(X)
            scores[item.key] = [accuracy_score(y,y_pred),f1_score(y,y_pred,average=average_f1)]

        # Save the baseline results if 'path' is not None
        if (not path is None):
            final_list = []
            for key in scores:
                aux = []
                model = key
                aux.append(model)
                arr = np.array(scores[key])

                for j in range(len(arr)):
                    aux.append(arr[j])

                final_list.append(aux)
            
            scores_baselines = np.array(final_list,dtype=object)
            np.savetxt('{}/results.txt'.format(path),scores_baselines,fmt='%s',delimiter=',',header='Model,Accuracy,F1-Score') 

        return scores        

    def save_clusters(self,path):
        '''
            Function that saves a text file with the clusters defined by the unsupervised OPF.

            Parameters
            ----------
            path: str
                Path to the folder where file will be saved.

            Returns
            -------
                None
        '''

        if (self.clusters is None):
            raise SystemExit('It is not possible to save the clusters because they are not defined yet! It might be happened because the model is not fitted yet!')
        
        import os

        # Auxiliary variables
        clusters = []
        prototypes = []

        for c in self.clusters:
            clusters.append([c,self.clusters[c]])

        for p in self.centroids_scores:
            prototypes.append([p,self.centroids_scores[p]])

        print('Saving the OPF clusters and their prototypes...')
        clusters = np.array(clusters,dtype=object)
        prototypes = np.array(prototypes,dtype=object)

        if (not os.path.exists(path)):
            os.makedirs(path)

        try:
            np.savetxt('{}/clusters.txt'.format(path),clusters,fmt='%s',delimiter=';',header='cluster_id,clusters')
            np.savetxt('{}/centroids.txt'.format(path),prototypes,fmt='%s',delimiter=',',header='centroid,F1')
        except Exception as e:
            raise SystemExit(f'Something went wrong while saving the clusters and their centroids! Error: {e}')