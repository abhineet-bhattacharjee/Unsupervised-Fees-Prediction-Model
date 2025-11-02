import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import os

from model_selection import RANDOM_STATE, SCHOOL_COLUMNS, X_scaled


os.makedirs('models', exist_ok=True)

def train_kmeans():
