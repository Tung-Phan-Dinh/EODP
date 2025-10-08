import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('../preprocess/trips_demographic.csv')

print("="*70)
print("LOGISTIC REGRESSION MULTI-CLASS CLASSIFICATION")
print("Transport Mode Prediction: Private, Public, Active, Other")
print("="*70)

print(f"\nOriginal dataset shape: {df.shape}")
print(f"\nTransport mode distribution:")
mode_counts = df['transport_mode'].value_counts()
print(mode_counts)
print("\nPercentage distribution:")
print((mode_counts / len(df) * 100).round(2))

# Select features as specified
features = ['hhsize', 'dwelltype', 'studying', 'carlicence', 'mainact', 'totalvehs']
target = 'transport_mode'

# Check for missing values
print(f"\nMissing values in features:")
print(df[features].isnull().sum())

# Remove rows with missing values in selected features or target
df_clean = df[features + [target]].dropna()

print(f"\nDataset shape after removing missing values: {df_clean.shape}")
print(f"\nFinal transport mode distribution:")
print(df_clean[target].value_counts())

# Prepare features and target
X = df_clean[features].copy()
y = df_clean[target].copy()

# Encode categorical variables
label_encoders = {}
categorical_features = ['dwelltype', 'studying', 'carlicence', 'mainact']

print("\n" + "="*70)
print("ENCODING CATEGORICAL FEATURES")
print("="*70)

for feature in categorical_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature].astype(str))
    label_encoders[feature] = le
    print(f"\n{feature} categories ({len(le.classes_)} unique values):")
    encoding_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    for cat, code in list(encoding_dict.items())[:10]:  # Show first 10
        print(f"  {cat}: {code}")
    if len(le.classes_) > 10:
        print(f"  ... and {len(le.classes_) - 10} more")

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "="*70)
print("TRAIN-TEST SPLIT")
print("="*70)
print(f"\nTraining set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Testing set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

print("\nTraining set class distribution:")
train_dist = y_train.value_counts()
print(train_dist)
print("\nTesting set class distribution:")
test_dist = y_test.value_counts()
print(test_dist)

# Standardize features (important for Logistic Regression)
print("\n" + "="*70)
print("FEATURE STANDARDIZATION")
print("="*70)
print("\nStandardizing features for logistic regression...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling complete!")
print("\nFeature means (after scaling):")
print(pd.DataFrame(X_train_scaled, columns=features).mean().round(4))
print("\nFeature standard deviations (after scaling):")
print(pd.DataFrame(X_train_scaled, columns=features).std().round(4))

# Create and train the Logistic Regression model
print("\n" + "="*70)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("="*70)

lr_model = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs',
    random_state=42,
    class_weight='balanced',  # Handle class imbalance
    C=1.0  # Regularization strength
)

print("\nModel Parameters:")
print(f"  - Solver: {lr_model.solver}")
print(f"  - Multi-class strategy: {lr_model.multi_class}")
print(f"  - Max iterations: {lr_model.max_iter}")
print(f"  - Regularization (C): {lr_model.C}")
print(f"  - Class weight: {lr_model.class_weight}")

print("\nTraining in progress...")
lr_model.fit(X_train_scaled, y_train)
print("Training complete!")
print(f"Number of iterations: {lr_model.n_iter_}")

# Make predictions
y_pred = lr_model.predict(X_test_scaled)
y_pred_proba = lr_model.predict_proba(X_test_scaled)

# Evaluate the model
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n" + "-"*70)
print("DETAILED CLASSIFICATION REPORT")
print("-"*70)
print(classification_report(y_test, y_pred))

print("\n" + "-"*70)
print("CONFUSION MATRIX")
print("-"*70)
cm = confusion_matrix(y_test, y_pred, labels=lr_model.classes_)
print(cm)
print(f"\nClasses order: {lr_model.classes_}")

# Per-class accuracy
print("\n" + "-"*70)
print("PER-CLASS PERFORMANCE")
print("-"*70)
for i, class_name in enumerate(lr_model.classes_):
    class_correct = cm[i, i]
    class_total = cm[i, :].sum()
    class_accuracy = class_correct / class_total if class_total > 0 else 0
    print(f"{class_name:12s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - "
          f"{class_correct}/{class_total} correct")

# Feature coefficients (importance)
print("\n" + "="*70)
print("FEATURE COEFFICIENTS")
print("="*70)

# Get coefficients for each class
coef_df = pd.DataFrame(
    lr_model.coef_.T,
    columns=lr_model.classes_,
    index=features
)

print("\nCoefficients for each class:")
print(coef_df.round(4))

# Calculate average absolute coefficient as feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'avg_abs_coefficient': np.abs(lr_model.coef_).mean(axis=0)
}).sort_values('avg_abs_coefficient', ascending=False)

print("\n" + "-"*70)
print("FEATURE IMPORTANCE (Average Absolute Coefficient)")
print("-"*70)
print(feature_importance.to_string(index=False))

# Visualizations
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# 1. Confusion Matrix
ax1 = fig.add_subplot(gs[0, :2])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=lr_model.classes_,
            yticklabels=lr_model.classes_,
            ax=ax1, cbar_kws={'label': 'Count'})
ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax1.set_title(f'Confusion Matrix\nOverall Accuracy: {accuracy:.4f}',
              fontsize=14, fontweight='bold')

# 2. Feature Importance (Average Absolute Coefficient)
ax2 = fig.add_subplot(gs[0, 2])
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
ax2.barh(feature_importance['feature'], feature_importance['avg_abs_coefficient'],
         color=colors)
ax2.set_xlabel('Avg Abs Coefficient', fontsize=11, fontweight='bold')
ax2.set_title('Feature Importance', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
for i, v in enumerate(feature_importance['avg_abs_coefficient']):
    ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

# 3. Class Distribution
ax3 = fig.add_subplot(gs[1, 0])
class_dist = y.value_counts().sort_index()
bars = ax3.bar(class_dist.index, class_dist.values,
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.7)
ax3.set_xlabel('Transport Mode', fontsize=11, fontweight='bold')
ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
ax3.set_title('Dataset Class Distribution', fontsize=12, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=9)

# 4. Per-Class Accuracy
ax4 = fig.add_subplot(gs[1, 1])
class_accuracies = []
for i, class_name in enumerate(lr_model.classes_):
    class_correct = cm[i, i]
    class_total = cm[i, :].sum()
    class_acc = class_correct / class_total if class_total > 0 else 0
    class_accuracies.append(class_acc)

bars = ax4.bar(lr_model.classes_, class_accuracies,
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.7)
ax4.set_xlabel('Transport Mode', fontsize=11, fontweight='bold')
ax4.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax4.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 1])
ax4.axhline(y=accuracy, color='red', linestyle='--', label='Overall Accuracy', linewidth=2)
ax4.tick_params(axis='x', rotation=45)
ax4.legend()
for bar, acc in zip(bars, class_accuracies):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{acc:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# 5. Prediction Distribution
ax5 = fig.add_subplot(gs[1, 2])
pred_dist = pd.Series(y_pred).value_counts().sort_index()
width = 0.35
x = np.arange(len(lr_model.classes_))
bars1 = ax5.bar(x - width/2, [test_dist.get(c, 0) for c in lr_model.classes_],
                width, label='Actual', alpha=0.8)
bars2 = ax5.bar(x + width/2, [pred_dist.get(c, 0) for c in lr_model.classes_],
                width, label='Predicted', alpha=0.8)
ax5.set_xlabel('Transport Mode', fontsize=11, fontweight='bold')
ax5.set_ylabel('Count', fontsize=11, fontweight='bold')
ax5.set_title('Actual vs Predicted (Test Set)', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(lr_model.classes_, rotation=45)
ax5.legend()

plt.savefig('logistic_regression_multiclass_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'logistic_regression_multiclass_results.png'")

# Create separate figure for Feature Coefficients by Class
fig2 = plt.figure(figsize=(10, 6))
sns.heatmap(coef_df.T, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Coefficient Value'})
plt.xlabel('Features', fontsize=12, fontweight='bold')
plt.ylabel('Transport Mode', fontsize=12, fontweight='bold')
plt.title('Feature Coefficients by Class\n(Positive values increase probability of that class)',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('logistic_regression_feature_coefficients.png', dpi=300, bbox_inches='tight')
print("Feature coefficients visualization saved as 'logistic_regression_feature_coefficients.png'")

# Example predictions
print("\n" + "="*70)
print("EXAMPLE PREDICTIONS")
print("="*70)

n_examples = 5
for i in range(min(n_examples, len(X_test))):
    print(f"\nExample {i+1}:")
    print(f"  Actual: {y_test.iloc[i]}")
    print(f"  Predicted: {y_pred[i]}")
    print(f"  Probabilities:")
    for j, class_name in enumerate(lr_model.classes_):
        print(f"    {class_name}: {y_pred_proba[i][j]:.4f} ({y_pred_proba[i][j]*100:.2f}%)")
    print(f"  Result: {'✓ CORRECT' if y_pred[i] == y_test.iloc[i] else '✗ INCORRECT'}")

# Model interpretation
print("\n" + "="*70)
print("MODEL INTERPRETATION")
print("="*70)
print("\nKey insights from coefficients:")
for class_name in lr_model.classes_:
    print(f"\n{class_name}:")
    class_coefs = coef_df[class_name].sort_values(ascending=False)
    print(f"  Top positive features:")
    for feat, coef in class_coefs.head(3).items():
        print(f"    - {feat}: {coef:.4f}")
    print(f"  Top negative features:")
    for feat, coef in class_coefs.tail(3).items():
        print(f"    - {feat}: {coef:.4f}")

print("\n" + "="*70)
print("MODEL TRAINING AND EVALUATION COMPLETE!")
print("="*70)
print(f"\nFinal Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Total samples: {len(df_clean)}")
print(f"Features used: {', '.join(features)}")
print(f"Classes predicted: {', '.join(lr_model.classes_)}")
print(f"\nModel type: Logistic Regression (Multinomial)")
print(f"Convergence: {lr_model.n_iter_[0]} iterations")