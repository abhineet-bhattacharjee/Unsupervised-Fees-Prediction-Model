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

results = []

def kmeans_clustering():
    for n_clusters in [2, 3, 4, 5, 6, 7]:
        start_time = time.time()

        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20, max_iter=300)
        labels = kmeans.fit_predict(X_scaled)

        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        inertia = kmeans.inertia_

        elapsed = time.time() - start_time

        results.append({
            'Model': 'K-Means',
            'Config': f'n_clusters={n_clusters}',
            'n_clusters': n_clusters,
            'Silhouette': round(silhouette, 4),
            'Davies_Bouldin': round(davies_bouldin, 4),
            'Calinski_Harabasz': round(calinski, 2),
            'Inertia': round(inertia, 4),
            'Time_sec': round(elapsed, 3)
        })

        print(f"k={n_clusters}: Silhouette={silhouette:.4f}, DB={davies_bouldin:.4f}, CH={calinski:.2f}, Time={elapsed:.2f}s")

def hierarchical_clustering():
    linkage_methods = ['ward', 'complete', 'average', 'single']
    for n_clusters in [2, 3, 4, 5]:
        for linkage in linkage_methods:
            start_time = time.time()

            agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            labels = agg.fit_predict(X_scaled)

            silhouette = silhouette_score(X_scaled, labels)
            davies_bouldin = davies_bouldin_score(X_scaled, labels)
            calinski = calinski_harabasz_score(X_scaled, labels)

            elapsed = time.time() - start_time

            results.append({
                'Model': 'Hierarchical',
                'Config': f'n={n_clusters}, link={linkage}',
                'n_clusters': n_clusters,
                'linkage': linkage,
                'Silhouette': round(silhouette, 4),
                'Davies_Bouldin': round(davies_bouldin, 4),
                'Calinski_Harabasz': round(calinski, 2),
                'Time_sec': round(elapsed, 3)
            })

            print(f"  n={n_clusters}, {linkage}: Silhouette={silhouette:.4f}, DB={davies_bouldin:.4f}")

