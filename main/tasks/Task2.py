import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import Task1

# Load and prepare data
products_df = pd.read_csv('Products.csv')
customers_df = pd.read_csv('Customers.csv')


def create_customer_features(customers_df):
    # Convert categorical variables to numeric
    region_encoded = pd.get_dummies(customers_df['Region'], prefix='Region')
    
    # Add signup date features
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    customers_df['AccountAge'] = (pd.Timestamp('2025-01-27') - customers_df['SignupDate']).dt.days
    
    # Combine features
    features = pd.concat([
        region_encoded,
        customers_df[['AccountAge']]
    ], axis=1)
    
    return features

def find_lookalikes(customer_features, customer_ids, target_ids, n_recommendations=3):
    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_features)
    
    # Calculate similarity matrix using cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(scaled_features)
    
    recommendations = []
    for target_id in target_ids:
        idx = customer_ids[customer_ids == target_id].index[0]
        similar_indices = similarity_matrix[idx].argsort()[::-1][1:n_recommendations+1]
        similar_scores = similarity_matrix[idx][similar_indices]
        
        recommendations.append({
            'customer_id': target_id,
            'similar_customers': [
                (customer_ids.iloc[idx], score) 
                for idx, score in zip(similar_indices, similar_scores)
            ]
        })
    
    return recommendations