import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

def classification_tree():
    """Build and evaluate a decision tree classifier for transport mode prediction"""

    # Load the data
    print("Loading data...")
    data = pd.read_csv('../preprocess/trips_demographic.csv')

    # Select features for classification
    feature_cols = ['hhsize', 'carlicence', 'studying', 'mainact', 'dwelltype']

    # Target variable
    target_col = 'transport_mode'

    # Drop rows with missing values
    data_clean = data[feature_cols + [target_col]].dropna()

    print(f"\nDataset size: {len(data_clean)} samples")
    print(f"Transport mode distribution:\n{data_clean[target_col].value_counts()}")

    # Encode categorical features
    print("\nEncoding categorical features...")
    label_encoders = {}
    X = data_clean[feature_cols].copy()

    for col in feature_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Encode target variable
    y = data_clean[target_col]

    # Split data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    # Build decision tree classifier
    print("\nTraining Decision Tree Classifier...")
    clf = DecisionTreeClassifier(
        max_depth=5,           # Limit tree depth to prevent overfitting
        min_samples_split=100, # Minimum samples required to split a node
        min_samples_leaf=50,   # Minimum samples required in a leaf node
        random_state=42
    )

    clf.fit(X_train, y_train)

    # Make predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Evaluate model
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("\n" + "="*80)
    print("MODEL PERFORMANCE")
    print("="*80)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy:  {test_accuracy:.4f}")

    # Classification report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT (Test Set)")
    print("="*80)
    print(classification_report(y_test, y_pred_test))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=clf.classes_,
                yticklabels=clf.classes_)
    plt.title('Confusion Matrix - Decision Tree Classifier')
    plt.ylabel('Actual Transport Mode')
    plt.xlabel('Predicted Transport Mode')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)
    print(feature_importance.to_string(index=False))

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance in Decision Tree')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Visualize the decision tree
    plt.figure(figsize=(20, 12))
    plot_tree(clf,
              feature_names=feature_cols,
              class_names=clf.classes_,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title('Decision Tree for Transport Mode Classification')
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "="*80)
    print("Visualizations saved:")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("  - decision_tree_visualization.png")
    print("="*80)

    return clf, X_test, y_test, y_pred_test, feature_importance


if __name__ == "__main__":
    classification_tree()
