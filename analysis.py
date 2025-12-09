import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def load_data(filepath):
    """Loads the dataset with correct encoding."""
    try:
        df = pd.read_csv(filepath, delimiter=';', encoding='ISO-8859-1')
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Error: File not found.")
        return None

def preprocess_data(df):
    """Preprocesses data for clustering."""
    features = df[['Student Country', 'Question Level', 'Topic', 'Type of Answer']]
    
    categorical_features = ['Student Country', 'Question Level', 'Topic']
    numerical_features = ['Type of Answer']
    
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ])
    
    return preprocessor, features

def perform_clustering(df, n_clusters=8):
    """Executes K-Means clustering pipeline."""
    print("Starting Clustering...")
    preprocessor, features = preprocess_data(df)
    
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('reduce', TruncatedSVD(n_components=2, random_state=42)),
        ('cluster', KMeans(n_clusters=n_clusters, random_state=42))
    ])
    
    pipeline.fit(features)
    labels = pipeline.named_steps['cluster'].labels_
    
    df['Cluster'] = labels
    print(f"Clustering complete. Assigned {n_clusters} clusters.")
    return df, pipeline

def perform_regression(df):
    """Performs Linear Regression to predict answer correctness."""
    print("Starting Regression Analysis...")
    
    # Feature Selection for Regression
    X = df[['Question ID', 'Student Country', 'Question Level', 'Topic']]
    y = df['Type of Answer']
    
    # Preprocessing
    cat_features = ['Student Country', 'Question Level', 'Topic']
    num_features = ['Question ID']
    
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(drop='first'), cat_features),
        ('num', 'passthrough', num_features)
    ])
    
    X_processed = preprocessor.fit_transform(X)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Regression Results - R²: {r2:.4f}, RMSE: {rmse:.4f}")
    return model

def main():
    # Load Data (Assuming MathE.csv is present)
    df = load_data('MathE.csv')
    
    if df is not None:
        # EDA Summary
        print("\nDataset Info:")
        print(df.info())
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        # Clustering
        df_clustered, cluster_model = perform_clustering(df)
        
        # Regression
        reg_model = perform_regression(df)
        
        print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
