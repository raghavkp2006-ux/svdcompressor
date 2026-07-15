import os
import time
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

def main():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "algorithm_dataset.csv")
    model_path = os.path.abspath(os.path.join(base_dir, "..", "models", "algorithm_agent.pkl"))
    
    if not os.path.exists(csv_path) or not os.path.exists(model_path):
        print("Missing dataset or model.")
        return

    df = pd.read_csv(csv_path).dropna()
    X = df.drop('best_algorithm', axis=1)
    y = df['best_algorithm']
    
    # Needs to match training split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = joblib.load(model_path)
    
    # 1. Predictions & Metrics
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Get unique classes in y_test and y_pred to ensure correct labels
    classes = sorted(list(set(y_test) | set(y_pred)))
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    
    print("=== Evaluation Metrics ===")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Macro Precision:  {precision:.4f}")
    print(f"Macro Recall:     {recall:.4f}")
    print(f"Macro F1-Score:   {f1:.4f}")
    print("\nClass Distribution (Entire Dataset):")
    print(y.value_counts().to_string())
    
    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(base_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"\nSaved confusion matrix to {cm_path}")
    
    # 3. Feature Importance Plot
    rf_model = pipeline.named_steps['rf']
    importances = rf_model.feature_importances_
    features = X.columns
    
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(8, 5))
    plt.title("Random Forest Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    fi_path = os.path.join(base_dir, "feature_importance.png")
    plt.savefig(fi_path, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance to {fi_path}")
    
    # 4. Mean Inference Time
    # Warmup
    _ = pipeline.predict(X_test.iloc[[0]])
    
    times = []
    for i in range(len(X_test)):
        sample = X_test.iloc[[i]]
        start = time.perf_counter()
        _ = pipeline.predict(sample)
        times.append(time.perf_counter() - start)
        
    mean_time_ms = np.mean(times) * 1000
    print(f"\nMean Inference Time (per image): {mean_time_ms:.4f} ms")

if __name__ == "__main__":
    main()
