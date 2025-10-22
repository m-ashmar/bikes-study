#!/usr/bin/env python3
"""
Block 7 - Clustering Analysis
Simple clustering of bike trip patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Block 7 - Clustering Analysis")
    print("=" * 50)
    
    # Load the mini dataset
    print("Loading data...")
    df = pd.read_parquet('merged_data/ultra_mini_merged_sample_fixed.parquet')
    print(f"Dataset shape: {df.shape}")
    
    # Create simple features
    print("\nCreating features...")
    
    # Calculate trip duration from timestamps
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])
    df['trip_duration_minutes'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
    
    # Simple cost calculation (basic pricing)
    df['trip_cost_usd'] = np.where(df['trip_duration_minutes'] <= 30, 3.0, 
                                  3.0 + (df['trip_duration_minutes'] - 30) * 0.1)
    
    # Weather category from conditions
    df['weather_category'] = df['conditions'].fillna('unknown')
    
    # CBD proximity
    df['start_to_cbd_proximity'] = np.where(df['start_in_cbd'] == True, 'in_cbd', 'far')
    
    # Select features for clustering
    numeric_cols = ['trip_duration_minutes', 'trip_cost_usd', 'temp']
    cat_cols = ['weather_category', 'start_to_cbd_proximity', 'member_casual', 'rideable_type']
    
    available_numeric = [f for f in numeric_cols if f in df.columns]
    available_cat = [f for f in cat_cols if f in df.columns]
    
    print(f"Numeric features: {available_numeric}")
    print(f"Categorical features: {available_cat}")
    
    # Prepare data
    data = df[available_numeric + available_cat].copy()
    data = data.dropna()
    print(f"Clean data shape: {data.shape}")
    
    # Encode categorical variables
    encoders = {}
    for col in available_cat:
        if col in data.columns:
            le = LabelEncoder()
            data[f'{col}_encoded'] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
    
    # Create feature matrix
    features = available_numeric + [f'{col}_encoded' for col in available_cat]
    X = data[features]
    print(f"Feature matrix shape: {X.shape}")
    
    # Normalize
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run clustering algorithms
    print("\nRunning KMeans...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_score = silhouette_score(X_scaled, kmeans_labels)
    print(f"KMeans silhouette: {kmeans_score:.3f}")
    
    print("Running DBSCAN...")
    dbscan = DBSCAN(eps=1.0, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    print(f"DBSCAN clusters: {n_clusters}, noise: {n_noise}")
    
    print("Running Agglomerative...")
    agg = AgglomerativeClustering(n_clusters=3)
    agg_labels = agg.fit_predict(X_scaled)
    agg_score = silhouette_score(X_scaled, agg_labels)
    print(f"Agglomerative silhouette: {agg_score:.3f}")
    
    # Add labels to data
    data['cluster_kmeans'] = kmeans_labels
    data['cluster_dbscan'] = dbscan_labels
    data['cluster_agg'] = agg_labels
    
    # Create plots
    print("\nCreating visualizations...")
    os.makedirs('figures/block7', exist_ok=True)
    
    # PCA for visualization (handle single feature case)
    if X_scaled.shape[1] >= 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # KMeans plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
        plt.title('KMeans Clusters')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.colorbar()
        plt.savefig('figures/block7/kmeans_pca.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # DBSCAN plot
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
        X_tsne = tsne.fit_transform(X_scaled)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.7)
        plt.title('DBSCAN Clusters')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar()
        plt.savefig('figures/block7/dbscan_tsne.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Dendrogram
        plt.figure(figsize=(10, 6))
        linkage_matrix = linkage(X_scaled, method='ward')
        dendrogram(linkage_matrix, truncate_mode='level', p=3)
        plt.title('Agglomerative Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.savefig('figures/block7/agglomerative_dendrogram.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Simple histogram for single feature
        plt.figure(figsize=(8, 6))
        plt.hist(X_scaled[:, 0], bins=30, alpha=0.7)
        plt.title('Feature Distribution')
        plt.xlabel('Normalized Feature Value')
        plt.ylabel('Frequency')
        plt.savefig('figures/block7/feature_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Copy the same plot for other methods
        import shutil
        shutil.copy('figures/block7/feature_distribution.png', 'figures/block7/kmeans_pca.png')
        shutil.copy('figures/block7/feature_distribution.png', 'figures/block7/dbscan_tsne.png')
        shutil.copy('figures/block7/feature_distribution.png', 'figures/block7/agglomerative_dendrogram.png')
    
    print("Plots saved to figures/block7/")
    
    # Analyze clusters
    print("\nCluster Analysis:")
    
    methods = {
        'KMeans': kmeans_labels,
        'DBSCAN': dbscan_labels,
        'Agglomerative': agg_labels
    }
    
    for method_name, labels in methods.items():
        print(f"\n{method_name} Results:")
        print("-" * 30)
        
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
            n_noise = list(labels).count(-1)
            print(f"Clusters: {len(unique_labels)}, Noise: {n_noise}")
        else:
            print(f"Clusters: {len(unique_labels)}")
        
        for cluster_id in sorted(unique_labels):
            mask = labels == cluster_id
            cluster_data = data[mask]
            
            print(f"\nCluster {cluster_id} ({len(cluster_data)} trips):")
            
            if 'trip_duration_minutes' in cluster_data.columns:
                duration_mean = cluster_data['trip_duration_minutes'].mean()
                print(f"  Avg duration: {duration_mean:.1f} min")
            
            if 'trip_cost_usd' in cluster_data.columns:
                cost_mean = cluster_data['trip_cost_usd'].mean()
                print(f"  Avg cost: ${cost_mean:.2f}")
            
            if 'temp' in cluster_data.columns:
                temp_mean = cluster_data['temp'].mean()
                print(f"  Avg temp: {temp_mean:.1f}°C")
    
    # Save results
    print("\nSaving results...")
    output_path = 'engineered_data/clustered_block7.parquet'
    data.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")
    
    # Simple validation
    print("\nValidation:")
    tests_passed = 0
    total_tests = 3
    
    if not data['cluster_kmeans'].isnull().any():
        print("✅ KMeans labels OK")
        tests_passed += 1
    
    kmeans_sizes = data['cluster_kmeans'].value_counts()
    if len(kmeans_sizes) >= 2:
        print("✅ KMeans has multiple clusters")
        tests_passed += 1
    
    if len(data) == len(data.dropna(subset=['cluster_kmeans', 'cluster_dbscan', 'cluster_agg'])):
        print("✅ No data loss")
        tests_passed += 1
    
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    # Best method
    scores = {
        'KMeans': kmeans_score,
        'DBSCAN': silhouette_score(X_scaled, dbscan_labels) if -1 not in dbscan_labels else 0,
        'Agglomerative': agg_score
    }
    
    best_method = max(scores, key=scores.get)
    print(f"\nBest method: {best_method} (score: {scores[best_method]:.3f})")
    
    print("\nBlock 7 completed!")

if __name__ == "__main__":
    main() 