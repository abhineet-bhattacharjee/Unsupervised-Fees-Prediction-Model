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


def train_pca():
    n_components = 3
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    X_reconstructed = pca.inverse_transform(X_pca)
    reconstruction_mse = np.mean((X_scaled - X_reconstructed) ** 2)
    explained_variance = np.sum(pca.explained_variance_ratio_)

    print(f"\nPCA (n={n_components})")
    print(f"  Variance Explained: {explained_variance:.4%}")
    print(f"  Reconstruction MSE: {reconstruction_mse:.6f}")

    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i + 1}: {var:.4%}")

    joblib.dump(pca, 'models/pca.pkl')
    return pca, X_pca


def save_results(kmeans_labels, X_pca):
    results_df = pd.DataFrame({
        'Year': df['academic.year'],
        'Cluster': kmeans_labels,
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2]
    })

    print(f"\nSample Results (first 5 years):")
    print(results_df.head().to_string(index=False))
    print("...")
    print(results_df.tail(3).to_string(index=False))

    return results_df


if __name__ == "__main__":
    print("Training K-Means Clustering...")
    kmeans, kmeans_labels = train_kmeans()

    print()

    print("Training PCA Dimensionality Reduction...")
    pca, X_pca = train_pca()

    print()

    results_df = save_results(kmeans_labels, X_pca)

    joblib.dump(scaler, 'models/scaler.pkl')

    print(f"\nModels saved:")
    print("models/kmeans.pkl")
    print("models/pca.pkl")
    print("models/scaler.pkl")