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
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    calinski = calinski_harabasz_score(X_scaled, labels)

    print(f"K-Means (k={n_clusters})")
    print(f"  Silhouette: {silhouette:.4f}")
    print(f"  Davies-Bouldin: {davies_bouldin:.4f}")
    print(f"  Calinski-Harabasz: {calinski:.2f}")

    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        years = df['academic.year'][labels == cluster_id].values
        print(f"  Cluster {cluster_id}: {count} years ({years[0]}-{years[-1]})")

    joblib.dump(kmeans, 'models/kmeans.pkl')
    return kmeans, labels