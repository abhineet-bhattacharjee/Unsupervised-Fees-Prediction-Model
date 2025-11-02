import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
import time


warnings.filterwarnings('ignore')

RANDOM_STATE = 42
SCHOOL_COLUMNS = [
    'Business (MBA)', 'Design', 'Divinity', 'Education',
    'GSAS', 'Government', 'Law', 'Medical/Dental',
    'Public Health (1-Year MPH)'
]

df = pd.read_csv('dataset.csv')
X = df[SCHOOL_COLUMNS].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)