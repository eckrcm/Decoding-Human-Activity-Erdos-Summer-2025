"""


To import copy paste below:

import os
import sys

# Function to Traverse up until you find the project root
def get_project_root(target="Decoding-Human-Activity-Erdos-Summer-2025"): # replace target name with the name of the root directory/repository name
    path = os.getcwd()
    while os.path.basename(path) != target:
        new_path = os.path.dirname(path)
        if new_path == path:
            raise FileNotFoundError(f"Could not find project root '{target}'.")
        path = new_path
    return path


project_root = get_project_root()
directory_path = os.path.join(project_root, "Python Files") 

# Add to Python path
if directory_path not in sys.path:
    sys.path.append(directory_path)

from utils import *  # * import all or specific functions
"""

import os
from collections import defaultdict, Counter
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report



import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from collections import defaultdict
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.decomposition   import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def loo_euclidean_1nn_accuracy(X, y):
    """
    Compute leave-one-out 1-NN accuracy using Euclidean distance.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Class labels.

    Returns
    -------
    acc : float
        Fraction of samples correctly classified by 1-NN under LOO.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    # compute full pairwise distance matrix
    D = squareform(pdist(X, metric='euclidean'))
    
    correct = 0
    for i in range(n):
        # ignore self-distance
        row = D[i].copy()
        row[i] = np.inf
        nn = row.argmin()
        if y[nn] == y[i]:
            correct += 1
    
    return correct / n


uci_test = pd.read_csv('./Data/UCI HAR Data Frame/uci_test.csv')
uci_train = pd.read_csv('./Data/UCI HAR Data Frame/uci_train.csv')
uci_df = pd.concat([uci_train,uci_test],ignore_index = True)

uci_df