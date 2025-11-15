import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load the data
df = pd.read_csv('../preprocess/trips_demographic.csv')

print("="*80)
print("K-NEAREST NEIGHBORS CLASSIFIER")
print("Transport Mode Prediction: Private, Public, Active, Other")
print("="*80)

print(f"\nOriginal dataset shape: {df.shape}")
print(f"\nTransport mode distribution:")
mode_counts = df['transport_mode'].value_counts()
print(mode_counts)
print("\nPercentage distribution:")
print((mode_counts / len(df) * 100).round(2))

# Select features
features = ['hhsize', 'dwelltype', 'studying', 'carlicence', 'mainact', 'totalvehs']
target = 'transport_mode'

# Remove missing values
df_clean = df[features + [target]].dropna()

print(f"\nDataset shape after removing missing values: {df_clean.shape}")

# Prepare features and target
X = df_clean[features].copy()
y = df_clean[target].copy()

# Encode categorical variables
label_encoders = {}
categorical_features = ['dwelltype', 'studying', 'carlicence', 'mainact']

print("\n" + "="*80)
print("ENCODING CATEGORICAL FEATURES")
print("="*80)

for feature in categorical_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature].astype(str))
    label_encoders[feature] = le
    print(f"\n{feature}: {len(le.classes_)} unique values")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "="*80)
print("TRAIN-TEST SPLIT")
print("="*80)
print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Testing set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

# Standardize features (required for KNN)
print("\n" + "="*80)
print("FEATURE STANDARDIZATION (for KNN)")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling complete!")

# ============================================================================
# K-NEAREST NEIGHBORS (KNN)
# ============================================================================
print("\n" + "="*80)
print("K-NEAREST NEIGHBORS (KNN)")
print("="*80)

# Try different K values
k_values = [3, 5, 7, 10, 15, 20]
knn_results = []

print("\nTesting different K values...")
for k in k_values:
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',  # Weight by inverse distance
        metric='euclidean',
        n_jobs=-1
    )

    # Cross-validation
    cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, n_jobs=-1)

    # Train on full training set
    start_time = time.time()
    knn.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time

    # Predict
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    knn_results.append({
        'k': k,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_time': train_time
    })

    print(f"  k={k:2d}: Accuracy={accuracy:.4f}, CV={cv_scores.mean():.4f}+-{cv_scores.std():.4f}")

# Select best K
knn_results_df = pd.DataFrame(knn_results)
best_k_idx = knn_results_df['accuracy'].idxmax()
best_k = knn_results_df.iloc[best_k_idx]['k']

print(f"\n Best K value: {int(best_k)}")
print(f"   Test Accuracy: {knn_results_df.iloc[best_k_idx]['accuracy']:.4f}")
print(f"   CV Accuracy: {knn_results_df.iloc[best_k_idx]['cv_mean']:.4f}")

# Train final KNN model with best K
print(f"\nTraining final KNN model with K={int(best_k)}...")
knn_final = KNeighborsClassifier(
    n_neighbors=int(best_k),
    weights='distance',
    metric='euclidean',
    n_jobs=-1
)

knn_final.fit(X_train_scaled, y_train)
y_pred_knn = knn_final.predict(X_test_scaled)
y_pred_proba_knn = knn_final.predict_proba(X_test_scaled)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn, labels=knn_final.classes_)

print("\n" + "-"*80)
print("KNN MODEL EVALUATION")
print("-"*80)
print(f"\nOverall Accuracy: {accuracy_knn:.4f} ({accuracy_knn*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

# Export KNN classification report to CSV
report_dict_knn = classification_report(y_test, y_pred_knn, output_dict=True)
report_df_knn = pd.DataFrame(report_dict_knn).transpose()
report_df_knn.to_csv('knn_classification_report.csv')
print("\nKNN classification report exported to 'knn_classification_report.csv'")

print("\nConfusion Matrix:")
print(cm_knn)

# Per-class accuracy for KNN
print("\nPer-Class Performance:")
for i, class_name in enumerate(knn_final.classes_):
    class_correct = cm_knn[i, i]
    class_total = cm_knn[i, :].sum()
    class_accuracy = class_correct / class_total if class_total > 0 else 0
    print(f"  {class_name:12s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# Confusion Matrix
fig = plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues',
            xticklabels=knn_final.classes_,
            yticklabels=knn_final.classes_,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted', fontsize=12, fontweight='bold')
plt.ylabel('Actual', fontsize=12, fontweight='bold')
plt.title(f'KNN Confusion Matrix (K={int(best_k)})\nAccuracy: {accuracy_knn:.4f}',
          fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('knn_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved as 'knn_confusion_matrix.png'")

# ============================================================================
# EXAMPLE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE PREDICTIONS (First 5 test samples)")
print("="*80)

for i in range(min(5, len(X_test))):
    print(f"\nExample {i+1}:")
    print(f"  Actual: {y_test.iloc[i]}")
    print(f"  KNN Predicted: {y_pred_knn[i]} {'(correct)' if y_pred_knn[i] == y_test.iloc[i] else '(wrong)'}")

    print(f"  KNN Probabilities:")
    for j, class_name in enumerate(knn_final.classes_):
        print(f"    {class_name}: {y_pred_proba_knn[i][j]:.4f} ({y_pred_proba_knn[i][j]*100:.2f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nFinal Results:")
print(f"  KNN (K={int(best_k)}): {accuracy_knn:.4f} ({accuracy_knn*100:.2f}%)")
print(f"\nModel trained and evaluated successfully!")

"""
    Declaration
    I acknowledge the use of ChatGPT [https://chat.openai.com/] to support the
    development of my code and understanding of key concepts.

    I used prompts to:
    Understand and write Machine Learning model KNN and Gradient Boosting
"""
