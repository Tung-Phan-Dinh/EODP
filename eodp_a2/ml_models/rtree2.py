import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import randint
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('../preprocess/trips_demographic.csv')

print(f"Original dataset shape: {df.shape}")
print(f"\nTransport mode distribution (original):")
print(df['transport_mode'].value_counts())

# Filter for Private and Public transport modes only
df_filtered = df[df['transport_mode'].isin(['Active', 'Public'])].copy()

print(f"\nFiltered dataset shape: {df_filtered.shape}")
print(f"\nTransport mode distribution (filtered):")
print(df_filtered['transport_mode'].value_counts())

# Select features as specified
features = ['hhsize', 'dwelltype', 'studying', 'carlicence', 'mainact', 'totalvehs','totalbikes']
target = 'transport_mode'

# Check for missing values
print(f"\nMissing values in features:")
print(df_filtered[features].isnull().sum())

# Remove rows with missing values in selected features
df_clean = df_filtered[features + [target]].dropna()

# Handle non-numeric values in totalbikes and totalvehs
# Convert to numeric, coercing errors to NaN, then drop those rows
numeric_features = ['totalbikes', 'totalvehs']
for col in numeric_features:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Drop any remaining NaN values created by coercion
df_clean = df_clean.dropna()

print(f"\nDataset shape after removing missing values: {df_clean.shape}")

# Prepare features and target
X = df_clean[features].copy()
y = df_clean[target].copy()

# Encode categorical variables
label_encoders = {}
categorical_features = ['dwelltype', 'studying', 'carlicence', 'mainact']

for feature in categorical_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature].astype(str))
    label_encoders[feature] = le
    print(f"\n{feature} encoding:")
    print(dict(zip(le.classes_, le.transform(le.classes_))))

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Create and train the Random Forest model with Randomized Search
print("\n" + "="*50)
print("RANDOMIZED SEARCH FOR BEST PARAMETERS")
print("="*50)

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 25),
    'min_samples_split': randint(2, 50),
    'min_samples_leaf': randint(1, 20),
}

# Base model
rf_base = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

print("\nParameter distributions:")
print("  - n_estimators: 50 to 200")
print("  - max_depth: 5 to 25")
print("  - min_samples_split: 2 to 50")
print("  - min_samples_leaf: 1 to 20")
print("\nRunning randomized search with 15 iterations and 3-fold CV...")

# Randomized search
random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_distributions,
    n_iter=15,
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    scoring='accuracy'
)

random_search.fit(X_train, y_train)

# Get best model
rf_model = random_search.best_estimator_

print("\nBest Parameters Found:")
print(f"  - n_estimators: {rf_model.n_estimators}")
print(f"  - max_depth: {rf_model.max_depth}")
print(f"  - min_samples_split: {rf_model.min_samples_split}")
print(f"  - min_samples_leaf: {rf_model.min_samples_leaf}")
print(f"  - Best CV Score: {random_search.best_score_:.4f}")

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nFeature importance plot saved as 'feature_importance.png'")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Active', 'Public'], 
            yticklabels=['Active', 'Public'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix plot saved as 'confusion_matrix.png'")

# Example prediction
print("\n" + "="*50)
print("EXAMPLE PREDICTION")
print("="*50)
sample_input = X_test.iloc[0:1]
sample_prediction = rf_model.predict(sample_input)
sample_proba = rf_model.predict_proba(sample_input)

print(f"Sample input features:")
print(sample_input)
print(f"\nPredicted transport mode: {sample_prediction[0]}")
print(f"Prediction probabilities: Private={sample_proba[0][0]:.4f}, Public={sample_proba[0][1]:.4f}")
print(f"Actual transport mode: {y_test.iloc[0]}")

print("\n" + "="*50)
print("Model training complete!")
print("="*50)