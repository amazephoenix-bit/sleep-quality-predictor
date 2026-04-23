import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def load_and_prep_data(filepath):
    """Load dataset and split into features and targets."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Features and Targets
    X = df.drop(['sleep_quality_score', 'tired_next_day'], axis=1)
    y_reg = df['sleep_quality_score']
    y_clf = df['tired_next_day']
    
    return X, y_reg, y_clf

def build_pipeline(model):
    """Create a pipeline with one-hot encoding for categorical data."""
    categorical_features = ['bedtime_category']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], 
        remainder='passthrough'
    )

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

def train_models():
    """Train both Regression and Classification models."""
    data_path = 'DATA/sleep_data.csv'
    X, y_reg, y_clf = load_and_prep_data(data_path)
    
    # Split data
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    # 1. Regression Model (Quality Score)
    reg_model = build_pipeline(LinearRegression())
    reg_model.fit(X_train, y_reg_train)
    print(f"Regression R2 Score: {reg_model.score(X_test, y_reg_test):.4f}")

    # 2. Classification Model (Tiredness)
    clf_model = build_pipeline(LogisticRegression())
    clf_model.fit(X_train, y_clf_train)
    print(f"Classification Accuracy: {clf_model.score(X_test, y_clf_test):.4f}")

    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(reg_model, 'models/sleep_quality_regressor.pkl')
    joblib.dump(clf_model, 'models/tiredness_classifier.pkl')
    print("Models saved to 'models/' directory.")

if __name__ == "__main__":
    print("--- Starting Sleep Quality Predictor Training ---")
    try:
        train_models()
    except Exception as e:
        print(f"Error: {e}")