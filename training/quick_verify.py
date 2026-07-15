import os, time, pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

base_dir = r'c:\antigravity_projects\svd compressor\training'
df = pd.read_csv(os.path.join(base_dir, 'algorithm_dataset.csv')).dropna()

print('1. Class Distribution:')
print(df['best_algorithm'].value_counts())

model = joblib.load(r'c:\antigravity_projects\svd compressor\models\algorithm_agent.pkl')
print('\n2. Model Classes:')
print(model.named_steps['rf'].classes_)

X = df.drop('best_algorithm', axis=1)
y = df['best_algorithm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

print('\n3. Split & Evaluation:')
print(f'Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred, zero_division=0))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred, labels=model.named_steps['rf'].classes_))

print('\n4. Dataset Info:')
print(f'Training Set Size: {len(X_train)}')
print(f'Test Set Size: {len(X_test)}')
test_classes = y_test.unique()
print(f'Classes in Test Set: {len(test_classes)} ({test_classes})')
if len(test_classes) < len(model.named_steps['rf'].classes_):
    print('FLAG: Test set is missing some classes present in the model.')

print('\n5. PNG Files:')
print(f'Confusion Matrix PNG exists: {os.path.exists(os.path.join(base_dir, "confusion_matrix.png"))}')
print(f'Path: {os.path.join(base_dir, "confusion_matrix.png")}')
print(f'Feature Importance PNG exists: {os.path.exists(os.path.join(base_dir, "feature_importance.png"))}')
print(f'Path: {os.path.join(base_dir, "feature_importance.png")}')

print('\n6. Inference Time:')
_ = model.predict(X_test.iloc[[0]])
times = []
for i in range(len(X_test)):
    sample = X_test.iloc[[i]]
    start = time.perf_counter()
    _ = model.predict(sample)
    times.append(time.perf_counter() - start)
mean_time_ms = np.mean(times) * 1000
print(f'Mean Inference Time (per image): {mean_time_ms:.4f} ms')
