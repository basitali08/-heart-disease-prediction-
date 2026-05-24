# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

def load_data(filepath):
    '''Load the heart disease dataset'''
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    '''Preprocess the heart disease dataset for machine learning'''
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Identify column types
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Numerical preprocessing: impute missing values with median, then scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit the preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the preprocessor for later use
    os.makedirs('models', exist_ok=True)
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print(f'Training set shape: {X_train_processed.shape}')
    print(f'Test set shape: {X_test_processed.shape}')
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

if __name__ == '__main__':
    # Load data - corrected path
    df = load_data('data/raw/heart_disease.csv')
    print(f'Loaded dataset with shape: {df.shape}')
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    
    print('Preprocessed data saved to data/processed/')
    print('Preprocessor saved to models/preprocessor.pkl')