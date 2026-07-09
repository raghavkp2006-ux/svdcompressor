import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "algorithm_dataset.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("Dataset is empty.")
        return
        
    print(f"Loaded dataset with {len(df)} samples.")
    df = df.dropna()
    print(f"Dataset after dropna: {len(df)} samples.")
    
    # Check class distribution
    print("Class distribution:")
    print(df['best_algorithm'].value_counts())
    
    # Features and labels
    X = df.drop('best_algorithm', axis=1)
    y = df['best_algorithm']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline with scaler and random forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5))
    ])
    
    print("Training Random Forest Classifier...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    
    # Some classes might be missing in test set if dataset is small, so handle zero_division
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save model
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "algorithm_agent.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
