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

            print(f"n={n_clusters}, {linkage}: Silhouette={silhouette:.4f}, DB={davies_bouldin:.4f}")

def gaussian_mixture_model():
    covariance_types = ['full', 'tied', 'diag', 'spherical']
    for n_components in [2, 3, 4, 5]:
        for cov_type in covariance_types:
            start_time = time.time()

            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=cov_type,
                    random_state=RANDOM_STATE,
                    max_iter=200
                )
                gmm.fit(X_scaled)
                labels = gmm.predict(X_scaled)

                bic = gmm.bic(X_scaled)
                aic = gmm.aic(X_scaled)
                log_likelihood = gmm.score(X_scaled)

                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(X_scaled, labels)
                    davies_bouldin = davies_bouldin_score(X_scaled, labels)
                else:
                    silhouette = np.nan
                    davies_bouldin = np.nan

                elapsed = time.time() - start_time

                results.append({
                    'Model': 'GMM',
                    'Config': f'n={n_components}, cov={cov_type}',
                    'n_components': n_components,
                    'covariance': cov_type,
                    'Silhouette': round(silhouette, 4) if not np.isnan(silhouette) else None,
                    'Davies_Bouldin': round(davies_bouldin, 4) if not np.isnan(davies_bouldin) else None,
                    'BIC': round(bic, 2),
                    'AIC': round(aic, 2),
                    'LogLikelihood': round(log_likelihood, 4),
                    'Time_sec': round(elapsed, 3)
                })

                print(
                    f"n={n_components}, {cov_type}: BIC={bic:.2f}, AIC={aic:.2f}, Sil={silhouette:.4f if not np.isnan(silhouette) else 'N/A'}")

            except Exception as e:
                print(f"GMM n={n_components}, cov={cov_type} failed: {str(e)[:50]}")

def dbscan_clustering():
    eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    min_samples_values = [2, 3, 4, 5]

    for eps in eps_values:
        for min_samples in min_samples_values:
            start_time = time.time()

            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            elapsed = time.time() - start_time

            if n_clusters > 1 and n_noise < len(labels):
                mask = labels != -1
                if np.sum(mask) > 1:
                    try:
                        silhouette = silhouette_score(X_scaled[mask], labels[mask])
                        davies_bouldin = davies_bouldin_score(X_scaled[mask], labels[mask])
                    except:
                        silhouette = np.nan
                        davies_bouldin = np.nan
                else:
                    silhouette = np.nan
                    davies_bouldin = np.nan
            else:
                silhouette = np.nan
                davies_bouldin = np.nan

            results.append({
                'Model': 'DBSCAN',
                'Config': f'eps={eps}, min={min_samples}',
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'Silhouette': round(silhouette, 4) if not np.isnan(silhouette) else None,
                'Davies_Bouldin': round(davies_bouldin, 4) if not np.isnan(davies_bouldin) else None,
                'Time_sec': round(elapsed, 3)
            })

            sil_str = f"{silhouette:.4f}" if not np.isnan(silhouette) else "N/A"
            print(f"eps={eps}, min={min_samples}: clusters={n_clusters}, noise={n_noise}, Sil={sil_str}")

def principal_component_analysis():
    for n_components in [2, 3, 4, 5, 6]:
        start_time = time.time()

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        X_reconstructed = pca.inverse_transform(X_pca)
        reconstruction_mse = np.mean((X_scaled - X_reconstructed) ** 2)

        explained_variance = np.sum(pca.explained_variance_ratio_)

        elapsed = time.time() - start_time

        results.append({
            'Model': 'PCA',
            'Config': f'n_components={n_components}',
            'n_components': n_components,
            'Reconstruction_MSE': round(reconstruction_mse, 6),
            'Variance_Explained': round(explained_variance, 4),
            'Component_Variance': [round(v, 4) for v in pca.explained_variance_ratio_],
            'Time_sec': round(elapsed, 3)
        })

        print(f"n={n_components}: MSE={reconstruction_mse:.6f}, Variance={explained_variance:.2%}, Time={elapsed:.3f}s")

def independent_component_analysis():
    for n_components in [2, 3, 4, 5]:
        start_time = time.time()

        ica = FastICA(n_components=n_components, random_state=RANDOM_STATE, max_iter=500)
        X_ica = ica.fit_transform(X_scaled)

        X_reconstructed = ica.inverse_transform(X_ica)
        reconstruction_mse = np.mean((X_scaled - X_reconstructed) ** 2)

        elapsed = time.time() - start_time

        results.append({
            'Model': 'ICA',
            'Config': f'n_components={n_components}',
            'n_components': n_components,
            'Reconstruction_MSE': round(reconstruction_mse, 6),
            'Time_sec': round(elapsed, 3)
        })

        print(f"n={n_components}: MSE={reconstruction_mse:.6f}, Time={elapsed:.3f}s")



if  __name__ == "__main__":
    kmeans_clustering()
    hierarchical_clustering()
    gaussian_mixture_model()
    dbscan_clustering()
    principal_component_analysis()
    independent_component_analysis()
