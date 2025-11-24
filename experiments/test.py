import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.datasets import load_iris,load_breast_cancer,load_wine
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import sys

sys.path.append('.')

from kmeans_ensemble import KMeansSemble
from ensemble import Ensemble
from divergence_measures import disagreement_matrix,paired_q_matrix

from opfython.models.supervised import SupervisedOPF

np.set_printoptions(threshold=sys.maxsize)

import warnings
warnings.filterwarnings('ignore') 

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2)

# List of pre-defined classifiers
# You can build the OPFsemble with a list of classifiers or with an instance of the Ensemble class
classifiers_list = [KNeighborsClassifier(),ExtraTreesClassifier(n_estimators=50),GaussianNB(),SVC(kernel='linear'),SVC(kernel='rbf'),RandomForestClassifier()]

# Ensemble class instance to construct the OPFsemble
classifiers = Ensemble(n_models=len(classifiers_list),models=classifiers_list)

# Uncomment the line below if you want a list of pre-defined classifiers
#classifiers = classifiers_list

# Build the KMeansEnsemble instance
ens = KMeansSemble(ensemble=classifiers,divergence='disagreement',meta_data_mode='count_class',bootstrapping=False,random_state=None)

# Get the meta data
meta_X = ens.fit(X_train,y_train)
# Fit the model with the meta data in order to find the clusters of similar classifiers
ens.fit_meta_model(meta_X,n_clusters=5)
# Get the predictions
y_pred = ens.predict(X_test,voting='mode')

#print(ens.clusters)
print(ens.centroids)

for item in ens.ensemble.items:
	print(item.key)
#print(ens.get_scores_baselines(X_test,y_test))

print('Accuracy: ',accuracy_score(y_test,y_pred))