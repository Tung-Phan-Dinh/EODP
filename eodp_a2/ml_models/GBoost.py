import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load the data
df = pd.read_csv('../preprocess/trips_demographic.csv')

print("="*80)
print("GRADIENT BOOSTING CLASSIFIER")
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

# ============================================================================
# GRADIENT BOOSTING
# ============================================================================
print("\n" + "="*80)
print("GRADIENT BOOSTING CLASSIFIER")
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

# Export Gradient Boosting classification report to CSV
report_dict_gb = classification_report(y_test, y_pred_gb, output_dict=True)
report_df_gb = pd.DataFrame(report_dict_gb).transpose()
report_df_gb.to_csv('gradient_boosting_classification_report.csv')
print("\nGradient Boosting classification report exported to 'gradient_boosting_classification_report.csv'")

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
# VISUALIZATIONS
# ============================================================================

# Confusion Matrix
fig = plt.figure(figsize=(8, 6))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens',
            xticklabels=gb_final.classes_,
            yticklabels=gb_final.classes_,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted', fontsize=12, fontweight='bold')
plt.ylabel('Actual', fontsize=12, fontweight='bold')
plt.title(f'Gradient Boosting Confusion Matrix\nAccuracy: {accuracy_gb:.4f}',
          fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('gradient_boosting_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved as 'gradient_boosting_confusion_matrix.png'")

# ============================================================================
# EXAMPLE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE PREDICTIONS (First 5 test samples)")
print("="*80)

for i in range(min(5, len(X_test))):
    print(f"\nExample {i+1}:")
    print(f"  Actual: {y_test.iloc[i]}")
    print(f"  GB Predicted:  {y_pred_gb[i]} {'(correct)' if y_pred_gb[i] == y_test.iloc[i] else '(wrong)'}")

    print(f"  GB Probabilities:")
    for j, class_name in enumerate(gb_final.classes_):
        print(f"    {class_name}: {y_pred_proba_gb[i][j]:.4f} ({y_pred_proba_gb[i][j]*100:.2f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nFinal Results:")
print(f"  Gradient Boosting: {accuracy_gb:.4f} ({accuracy_gb*100:.2f}%)")
print(f"\nModel trained and evaluated successfully!")

"""
    Declaration
    I acknowledge the use of ChatGPT [https://chat.openai.com/] to support the
    development of my code and understanding of key concepts.

    I used prompts to:
    Understand and write Machine Learning model KNN and Gradient Boosting
"""
