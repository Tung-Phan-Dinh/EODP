import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load the data
df = pd.read_csv('../preprocess/trips_demographic.csv')

print("="*80)
print("K-NEAREST NEIGHBORS & GRADIENT BOOSTING CLASSIFIERS")
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
# MODEL 1: K-NEAREST NEIGHBORS (KNN)
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: K-NEAREST NEIGHBORS (KNN)")
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
    
    print(f"  k={k:2d}: Accuracy={accuracy:.4f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# Select best K
knn_results_df = pd.DataFrame(knn_results)
best_k_idx = knn_results_df['accuracy'].idxmax()
best_k = knn_results_df.iloc[best_k_idx]['k']

print(f"\n🏆 Best K value: {int(best_k)}")
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
# MODEL 2: GRADIENT BOOSTING
# ============================================================================
print("\n" + "="*80)
print("MODEL 2: GRADIENT BOOSTING CLASSIFIER")
print("="*80)

# Try different hyperparameters
gb_configs = [
    {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3},
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
    {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5},
    {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 5},
]

gb_results = []

print("\nTesting different hyperparameter configurations...")
print("(This may take a few minutes...)")

for idx, config in enumerate(gb_configs, 1):
    print(f"\nConfig {idx}/{len(gb_configs)}: {config}")
    
    gb = GradientBoostingClassifier(
        n_estimators=config['n_estimators'],
        learning_rate=config['learning_rate'],
        max_depth=config['max_depth'],
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )
    
    # Train
    start_time = time.time()
    gb.fit(X_train, y_train)  # GB doesn't need scaling
    train_time = time.time() - start_time
    
    # Predict
    y_pred = gb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    gb_results.append({
        'config': f"n={config['n_estimators']}, lr={config['learning_rate']}, depth={config['max_depth']}",
        'n_estimators': config['n_estimators'],
        'learning_rate': config['learning_rate'],
        'max_depth': config['max_depth'],
        'accuracy': accuracy,
        'train_time': train_time
    })
    
    print(f"  Accuracy: {accuracy:.4f}, Training time: {train_time:.2f}s")

# Select best configuration
gb_results_df = pd.DataFrame(gb_results)
best_gb_idx = gb_results_df['accuracy'].idxmax()
best_gb_config = gb_configs[best_gb_idx]

print(f"\n Best Gradient Boosting Configuration:")
print(f"   n_estimators: {best_gb_config['n_estimators']}")
print(f"   learning_rate: {best_gb_config['learning_rate']}")
print(f"   max_depth: {best_gb_config['max_depth']}")
print(f"   Test Accuracy: {gb_results_df.iloc[best_gb_idx]['accuracy']:.4f}")

# Train final GB model
print("\nTraining final Gradient Boosting model...")
gb_final = GradientBoostingClassifier(
    n_estimators=best_gb_config['n_estimators'],
    learning_rate=best_gb_config['learning_rate'],
    max_depth=best_gb_config['max_depth'],
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42,
    verbose=0
)

gb_final.fit(X_train, y_train)
y_pred_gb = gb_final.predict(X_test)
y_pred_proba_gb = gb_final.predict_proba(X_test)

accuracy_gb = accuracy_score(y_test, y_pred_gb)
cm_gb = confusion_matrix(y_test, y_pred_gb, labels=gb_final.classes_)

print("\n" + "-"*80)
print("GRADIENT BOOSTING MODEL EVALUATION")
print("-"*80)
print(f"\nOverall Accuracy: {accuracy_gb:.4f} ({accuracy_gb*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb))

print("\nConfusion Matrix:")
print(cm_gb)

# Per-class accuracy for GB
print("\nPer-Class Performance:")
for i, class_name in enumerate(gb_final.classes_):
    class_correct = cm_gb[i, i]
    class_total = cm_gb[i, :].sum()
    class_accuracy = class_correct / class_total if class_total > 0 else 0
    print(f"  {class_name:12s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")

# Feature importance for GB
print("\n" + "-"*80)
print("FEATURE IMPORTANCE (Gradient Boosting)")
print("-"*80)
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': gb_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\n", feature_importance.to_string(index=False))

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

comparison = pd.DataFrame({
    'Model': ['K-Nearest Neighbors', 'Gradient Boosting'],
    'Accuracy': [accuracy_knn, accuracy_gb],
    'Training Time (s)': [
        knn_results_df.iloc[best_k_idx]['train_time'],
        gb_results_df.iloc[best_gb_idx]['train_time']
    ]
})

print("\n", comparison.to_string(index=False))

if accuracy_gb > accuracy_knn:
    diff = (accuracy_gb - accuracy_knn) * 100
    print(f"\n Winner: Gradient Boosting (by {diff:.2f} percentage points)")
else:
    diff = (accuracy_knn - accuracy_gb) * 100
    print(f"\n Winner: K-Nearest Neighbors (by {diff:.2f} percentage points)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# Figure 1: Confusion Matrices
fig1 = plt.figure(figsize=(14, 6))

# 1. KNN Confusion Matrix
ax1 = fig1.add_subplot(1, 2, 1)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=knn_final.classes_,
            yticklabels=knn_final.classes_,
            cbar_kws={'label': 'Count'})
ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax1.set_title(f'KNN Confusion Matrix (K={int(best_k)})\nAccuracy: {accuracy_knn:.4f}',
              fontsize=14, fontweight='bold')

# 2. GB Confusion Matrix
ax2 = fig1.add_subplot(1, 2, 2)
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', ax=ax2,
            xticklabels=gb_final.classes_,
            yticklabels=gb_final.classes_,
            cbar_kws={'label': 'Count'})
ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax2.set_title(f'Gradient Boosting Confusion Matrix\nAccuracy: {accuracy_gb:.4f}',
              fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('knn_gb_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrices saved as 'knn_gb_confusion_matrices.png'")

# Figure 2: Model Accuracy and Per-Class Accuracy Comparison
fig2 = plt.figure(figsize=(14, 6))

# 1. Overall Model Accuracy Comparison
ax3 = fig2.add_subplot(1, 2, 1)
models = ['KNN', 'Gradient\nBoosting']
accuracies = [accuracy_knn, accuracy_gb]
colors_comp = ['#4ECDC4', '#FF6B6B']
bars = ax3.bar(models, accuracies, color=colors_comp, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax3.set_title('Overall Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax3.set_ylim([0, 1])
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{acc:.4f}\n({acc*100:.2f}%)', ha='center', va='bottom',
            fontsize=11, fontweight='bold')
ax3.grid(True, axis='y', alpha=0.3)

# 2. Per-Class Accuracy Comparison
ax4 = fig2.add_subplot(1, 2, 2)
classes = knn_final.classes_
knn_class_acc = []
gb_class_acc = []
for i in range(len(classes)):
    knn_class_acc.append(cm_knn[i, i] / cm_knn[i, :].sum() if cm_knn[i, :].sum() > 0 else 0)
    gb_class_acc.append(cm_gb[i, i] / cm_gb[i, :].sum() if cm_gb[i, :].sum() > 0 else 0)

x = np.arange(len(classes))
width = 0.35
bars1 = ax4.bar(x - width/2, knn_class_acc, width, label='KNN', alpha=0.8, color='#4ECDC4')
bars2 = ax4.bar(x + width/2, gb_class_acc, width, label='Gradient Boosting', alpha=0.8, color='#FF6B6B')
ax4.set_xlabel('Transport Mode', fontsize=12, fontweight='bold')
ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax4.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(classes, rotation=45)
ax4.set_ylim([0, 1])
ax4.legend(fontsize=11)
ax4.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('knn_gb_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("Accuracy comparison saved as 'knn_gb_accuracy_comparison.png'")

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
    print(f"  GB Predicted:  {y_pred_gb[i]} {'(correct)' if y_pred_gb[i] == y_test.iloc[i] else '(wrong)'}")
    
    print(f"  KNN Probabilities:")
    for j, class_name in enumerate(knn_final.classes_):
        print(f"    {class_name}: {y_pred_proba_knn[i][j]:.4f} ({y_pred_proba_knn[i][j]*100:.2f}%)")
    
    print(f"  GB Probabilities:")
    for j, class_name in enumerate(gb_final.classes_):
        print(f"    {class_name}: {y_pred_proba_gb[i][j]:.4f} ({y_pred_proba_gb[i][j]*100:.2f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nFinal Results:")
print(f"  KNN (K={int(best_k)}): {accuracy_knn:.4f} ({accuracy_knn*100:.2f}%)")
print(f"  Gradient Boosting: {accuracy_gb:.4f} ({accuracy_gb*100:.2f}%)")
print(f"\nBoth models trained and evaluated successfully!")