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

# Task 1: EDA and Business Insights

def perform_product_analysis(products_df):
    # Category analysis
    category_stats = products_df.groupby('Category').agg({
        'Price': ['count', 'mean', 'min', 'max']
    }).round(2)
    
    # Price distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Category', y='Price', data=products_df)
    plt.title('Price Distribution by Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('price_distribution.png')
    
    return category_stats

def perform_customer_analysis(customers_df):
    # Region distribution
    region_distribution = customers_df['Region'].value_counts()
    
    # Signup analysis
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    customers_df['SignupYear'] = customers_df['SignupDate'].dt.year
    signup_trends = customers_df.groupby(['SignupYear', 'Region']).size().unstack()
    
    # Visualize region distribution
    plt.figure(figsize=(8, 6))
    region_distribution.plot(kind='bar')
    plt.title('Customer Distribution by Region')
    plt.tight_layout()
    plt.savefig('region_distribution.png')
    
    return region_distribution, signup_trends
