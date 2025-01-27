
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load and prepare data
products_df = pd.read_csv('Products.csv')
customers_df = pd.read_csv('Customers.csv')
# Task 3: Customer Segmentation
def perform_clustering(customer_features, n_clusters_range=range(2, 11)):
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_features)
    
    # Find optimal number of clusters
    db_scores = []
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(scaled_features)
        db_score = davies_bouldin_score(scaled_features, labels)
        db_scores.append(db_score)
    
    # Get optimal number of clusters
    optimal_n_clusters = n_clusters_range[np.argmin(db_scores)]
    
    # Perform final clustering
    final_kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
    cluster_labels = final_kmeans.fit_predict(scaled_features)
    
    return cluster_labels, db_scores[np.argmin(db_scores)], optimal_n_clusters

def main():
    # Task 1: EDA
    print("Performing EDA...")
    category_stats = perform_product_analysis(products_df)
    region_dist, signup_trends = perform_customer_analysis(customers_df)
    
    print("\nProduct Category Statistics:")
    print(category_stats)
    print("\nCustomer Region Distribution:")
    print(region_dist)
    
    # Task 2: Lookalike Model
    print("\nGenerating Lookalike Recommendations...")
    customer_features = create_customer_features(customers_df)
    target_customers = customers_df['CustomerID'][:20]  # First 20 customers
    recommendations = find_lookalikes(customer_features, customers_df['CustomerID'], target_customers)
    
    # Save lookalike results
    lookalike_results = pd.DataFrame(recommendations)
    lookalike_results.to_csv('FirstName_LastName_Lookalike.csv', index=False)
    
    # Task 3: Customer Segmentation
    print("\nPerforming Customer Segmentation...")
    cluster_labels, db_score, optimal_clusters = perform_clustering(customer_features)
    
    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Davies-Bouldin Index: {db_score:.4f}")
    
    # Save clustering results
    clustering_results = pd.DataFrame({
        'CustomerID': customers_df['CustomerID'],
        'Cluster': cluster_labels
    })
    
    return {
        'category_stats': category_stats,
        'region_dist': region_dist,
        'signup_trends': signup_trends,
        'lookalike_results': lookalike_results,
        'clustering_results': clustering_results,
        'db_score': db_score
    }

if __name__ == "__main__":
    results = main()