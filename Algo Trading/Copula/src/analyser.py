import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    """Load and preprocess the data for clustering."""
    df = pd.read_csv(file_path)
    
    # Select features for clustering
    features = ['theta', 'psi', 'cvm_statistic', 
                'asset1_shape', 'asset1_scale', 'asset1_location',
                'asset2_shape', 'asset2_scale', 'asset2_location']
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    return df, scaled_features, features

def perform_clustering(scaled_features, n_clusters=3):
    """Perform K-means clustering on the scaled features."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    return clusters, kmeans

def analyze_clusters(df, clusters, features, n_clusters):
    """Analyze and visualize the clustering results."""
    # Add cluster labels to the original dataframe
    df['cluster'] = clusters
    
    # Save the clustered data to CSV
    output_file = f'analytics/clustered_data_{n_clusters}.csv'
    df.to_csv(output_file, index=False)
    print(f"\nClustered data saved to: {output_file}")
    
    # Print cluster statistics
    print("\nCluster Statistics:")
    print(df.groupby('cluster')[features].mean())
    
    # Visualize clusters using PCA for dimensionality reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(df[features])
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                         c=clusters, cmap='viridis')
    plt.title('Cluster Visualization (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(f'analytics/cluster_visualization_{n_clusters}.png')
    plt.show()
    
    # Plot feature distributions by cluster
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features):
        plt.subplot(3, 3, i+1)
        sns.boxplot(x='cluster', y=feature, data=df)
        plt.title(f'{feature} by Cluster')
    plt.tight_layout()
    plt.savefig(f'analytics/feature_distributions_{n_clusters}.png')
    plt.show()

def main():
    n_clusters = 2
    # Load and preprocess data
    df, scaled_features, features = load_and_preprocess_data('analytics/combined_summary.csv')
    
    # Perform clustering
    clusters, kmeans = perform_clustering(scaled_features, n_clusters)
    
    # Analyze and visualize results
    analyze_clusters(df, clusters, features, n_clusters)

if __name__ == "__main__":
    main()
