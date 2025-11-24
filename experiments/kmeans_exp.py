from time import time
from sklearn.metrics import accuracy_score,f1_score
from copy import deepcopy
import os
import sys
import numpy as np
import warnings
import argparse

sys.path.append('.')

from kmeans_ensemble import KMeansSemble
from ensemble import Ensemble

# Disable all types of warning
warnings.filterwarnings("ignore")

def get_main_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-i', type=str, help='Folder where the dataset is located.')
    parser.add_argument('-m', type=str, help='Folder where the base models are located.')
    parser.add_argument('-o', type=str, help='Folder where the results will be save.')

    return parser.parse_args()

def dict2array(dictionary):
    '''
        Function that transforms a dictionary into a numpy array in a following format:
        
        #################################
        #Model     Accuracy     F1-score
        Model_1    Acc_value1   F1_value1
        Model_2    Acc_value2   F1_value2
        .
        .
        .
        Model_n    Acc_valuen   F1_valuen
        #################################

        Parameters
        ----------
            dictionary: dict
                A dictionary in which the keys are the models' name, and the values are a list with two positions: Accuracy and F1 values
    '''

    final_list = []
    for key in dictionary:
        aux = []
        model = key
        aux.append(model)
        arr = np.array(dictionary[key])

        for j in range(len(arr)):
            aux.append(arr[j])

        final_list.append(aux)

    return np.array(final_list,dtype=object)

if __name__ == '__main__':
    args = get_main_args()
    data = args.i
    models_path = args.m
    results_folder = args.o

    # Check if the data folder exists
    if (not os.path.exists(data)):
        sys.exit('Data folder does not exists. Please check if the path to the data is correct or already exists.')

    # Reading the datasets' folders
    #ds = [d for d in os.listdir(data) if os.path.isdir('{}/{}'.format(data,d))]
    ds = ['iris']

    # Auxiliary variables
    folds = output_folder = y_pred = meta_X = None
    start_time_kmeans = end_time_kmeans = 0
    kmeans_variants = ['mode','intracluster',] # kmeanssemble variants
    divergence = {'nodivergence':None,'yule':'yule','disagreement':'disagreement'} # The divergence metrics
    meta_data_type = {'oracle':'oracle','countclass':'count_class'} # The meta-data type of the classifiers predictions
    n_folds_ensemble = 10 # Number of folds to construct the meta-data from the baseline classifiers

    n_clusters_list = [2,5,10,20,30] # List of k values to be tested to build the clusters
    kmeans_ens = KMeansSemble() # KMeansEnsemble instance

    # Performs the experiments for each meta data type
    for meta in meta_data_type:    
        # Performs the experiments for each divergence mode
        for dv in divergence:
            # Performs the experiments for each dataset
            for d in ds:
                # For each cross validation fold
                folds = os.listdir('{}/{}'.format(data,d))
                print('Number of folds for the dataset {}: {}'.format(d,len(folds)))

                for f in folds:
                    # Loading the training, validation and test sets
                    train = np.loadtxt('{}/{}/{}/train.txt'.format(data,d,f),delimiter=',')
                    valid = np.loadtxt('{}/{}/{}/valid.txt'.format(data,d,f),delimiter=',')
                    test = np.loadtxt('{}/{}/{}/test.txt'.format(data,d,f),delimiter=',')

                    # Split the data into X and y
                    # The last column is the output variable to be predicted
                    X_train,X_valid,X_test = train[:,:-1],valid[:,:-1],test[:,:-1]
                    y_train,y_valid,y_test = train[:,-1].astype(int),valid[:,-1].astype(int),test[:,-1].astype(int)

                    # For each number of baseline classifiers
                    for n in [10]:#,30,50]:
                        print('META TYPE {}, DATASET {}, FOLD {}, NUMBER OF CLASSIFIERS {} '.format(meta,d,f,n))
                        # Loading the baseline models
                        if (models_path != None):
                            print('Loading the baseline classifier models...')
                            ens = Ensemble(n_models=n,loading_path='{}/{}'.format(models_path,n))
                        else:
                            ens = None

                        # Checks if the model's results folders already exist
                        baseline_results = '{}/{}/{}/{}/{}/results.txt'.format(results_folder,'baseline',d,f,n)
                        
                        for v in kmeans_variants:
                            if (meta=='countclass'):
                                kmeanssemble_results = '{}/{}/{}/{}/{}/{}/results.txt'.format(results_folder,'kmeans_{}'.format(meta),v,d,f,n)
                            else:
                                kmeanssemble_results = '{}/{}/{}/{}/{}/{}/results.txt'.format(results_folder,'kmeans_{}'.format(dv),v,d,f,n)
                            
                        # Assigns the kmeanssemble's parameters
                        kmeans_ens.n_models=n
                        kmeans_ens.n_folds=n_folds_ensemble
                        kmeans_ens.divergence=divergence[dv]
                        kmeans_ens.ensemble = ens
                        kmeans_ens.meta_data_mode = meta_data_type[meta]

                        # Training the kmeanssemble
                        print('Building the meta-data of the kmeanssemble....')
                        start_time_kmeans = time()
                        meta_X = kmeans_ens.fit(X_train,y_train)
                        end_time_kmeans = time() -start_time_kmeans                

                        # Gets the baseline model's predictions
                        # Check if the baseline model's folder results already exists
                        output_folder = '{}/{}/{}/{}/{}'.format(results_folder,'baseline',d,f,n)
                        if (not os.path.exists(output_folder)):
                            os.makedirs(output_folder)
                        
                        if (not os.path.exists('{}/results.txt'.format(output_folder))):
                            print('Performing the test with the baseline models...')
                            # Getting the baseline predictions as array
                            scores_baselines = dict2array(kmeans_ens.get_scores_baselines(X_test,y_test))
                            # Saving the baseline's results y_pred
                            np.savetxt('{}/results.txt'.format(output_folder),scores_baselines,fmt='%s',delimiter=',',header='Model,Accuracy,F1')
                        else:
                            print('Folder {} already exists with all the validation metrics...'.format(output_folder))
                        
                        # Test with the kmeanssemble and its variants
                        for v in kmeans_variants:
                            print('KMeans variant: ',v)
                            # Check if the output folder exists
                            if (meta=='countclass'):
                                output_folder = '{}/{}/{}/{}/{}/{}'.format(results_folder,'kmeans_{}'.format(meta),v,d,f,n)
                            else:
                                output_folder = '{}/{}/{}/{}/{}/{}'.format(results_folder,'kmeans_{}'.format(dv),v,d,f,n)
                            if (not os.path.exists(output_folder)):
                                os.makedirs(output_folder)
                            
                            if(os.path.exists('{}/results.txt'.format(output_folder))):
                                print('Folder {} already exists. Moving to the next KMeans variant...'.format(output_folder))
                                continue

                            # Seeks the best value for k_max using the validation set
                            best_k_max = None
                            highest_f1 = -1
                            k_max_valid = []
                            for k_max in n_clusters_list:
                                start_time_unsup = time()
                                kmeans_ens.fit_meta_model(meta_X,k_max)
                                end_time_unsup = time() - start_time_unsup

                                y_pred = kmeans_ens.predict(X_valid,voting=v)
                                f1 = f1_score(y_valid,y_pred,average='weighted')

                                if (f1 > highest_f1):
                                    highest_f1 = f1
                                    best_k_max = k_max

                                k_max_valid.append([k_max,f1,end_time_unsup])

                            # Saving the tested n_clusters values and their F1 scores
                            np.savetxt('{}/n_clusters_validation.txt'.format(output_folder),np.array(k_max_valid),fmt='%.4f',delimiter=',',header='n_clusters,F1,Meta model time')

                            # KMeanssemble predictions using the test set and the best n_clusters value
                            kmeans_ens.fit_meta_model(meta_X,n_clusters=best_k_max)
                            y_pred = kmeans_ens.predict(X_test,voting=v)
                            # Computing accuracy and f1-score
                            acc = accuracy_score(y_test,y_pred)
                            f1 = f1_score(y_test,y_pred,average='weighted')
                            # Saving the validation measures, the meta_X and y_pred
                            np.savetxt('{}/y_pred.txt'.format(output_folder),y_pred,fmt='%.4f',delimiter=',')
                            np.savetxt('{}/meta_X.txt'.format(output_folder),meta_X,fmt='%.4f',delimiter=',')
                            np.savetxt('{}/results.txt'.format(output_folder),np.array([acc,f1,end_time_kmeans]),fmt='%.4f',delimiter=',',header='Accuracy,F1,Fit time')                    
                            # Saving the clusters and their prototypes
                            kmeans_ens.save_clusters(output_folder)