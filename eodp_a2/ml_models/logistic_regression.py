import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, cross_validate,
                                      StratifiedKFold, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_data():
    """Load and prepare data for Logistic Regression"""
    # Load the data
    data = pd.read_csv('../preprocess/trips_demographic.csv')

    # Select features based on correlation analysis
    categorical_features = ['hhsize', 'sex', 'carlicence', 'anywork', 'studying',
                           'mainact', 'dwelltype', 'owndwell', 'hhinc_category', 'persinc_category']

    # Encode categorical variables
    data_encoded = data.copy()
    for col in categorical_features:
        data_encoded[col + '_encoded'] = pd.factorize(data_encoded[col])[0]

    # Prepare feature matrix (X) and target variable (y)
    feature_cols = [col + '_encoded' for col in categorical_features]
    X = data_encoded[feature_cols].dropna()
    y = data_encoded.loc[X.index, 'transport_mode']

    print("="*80)
    print("DATA PREPARATION")
    print("="*80)
    print(f"Total samples: {len(X)}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"\nFeatures used: {categorical_features}")
    print(f"\nTransport mode distribution:")
    print(y.value_counts())
    print(f"\nClass proportions:")
    print(y.value_counts(normalize=True))

    return X, y, categorical_features


def train_logistic_regression(X, y, test_size=0.2, random_state=42):
    """Train Logistic Regression classifier with feature scaling"""
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("\n" + "="*80)
    print("TRAINING LOGISTIC REGRESSION CLASSIFIER")
    print("="*80)
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Feature scaling (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize Logistic Regression
    # For multiclass: solver='lbfgs' or 'saga', max_iter increased for convergence
    lr_classifier = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced',  # Handle class imbalance
        n_jobs=-1
    )

    # Train the model
    print("\nTraining model...")
    lr_classifier.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_train = lr_classifier.predict(X_train_scaled)
    y_pred_test = lr_classifier.predict(X_test_scaled)

    # Get prediction probabilities
    y_pred_proba_test = lr_classifier.predict_proba(X_test_scaled)

    # Evaluate performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return lr_classifier, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_pred_test, y_pred_proba_test


def evaluate_model(lr_classifier, X_test, y_test, y_pred_test):
    """Evaluate the Logistic Regression model"""
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=lr_classifier.classes_,
                yticklabels=lr_classifier.classes_)
    plt.title('Confusion Matrix - Logistic Regression')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix_lr.png', dpi=300, bbox_inches='tight')
    plt.show()

    return cm


def feature_coefficients(lr_classifier, feature_names):
    """Analyze and visualize feature coefficients"""
    print("\n" + "="*80)
    print("FEATURE COEFFICIENTS ANALYSIS")
    print("="*80)

    # Get coefficients for each class
    classes = lr_classifier.classes_
    coefficients = lr_classifier.coef_

    print(f"\nNumber of classes: {len(classes)}")
    print(f"Coefficient matrix shape: {coefficients.shape}")

    # Create visualization for each class
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, 1, figsize=(12, 4*n_classes))

    if n_classes == 1:
        axes = [axes]

    for idx, (class_name, coef) in enumerate(zip(classes, coefficients)):
        # Create dataframe for this class
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coef
        }).sort_values('Coefficient', key=abs, ascending=False)

        print(f"\n{class_name} - Top 5 Features:")
        print(coef_df.head().to_string(index=False))

        # Plot coefficients
        ax = axes[idx]
        colors = ['red' if c < 0 else 'green' for c in coef_df['Coefficient']]
        ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
        ax.set_xlabel('Coefficient Value')
        ax.set_ylabel('Feature')
        ax.set_title(f'Feature Coefficients for {class_name}')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('feature_coefficients_lr.png', dpi=300, bbox_inches='tight')
    plt.show()


def stratified_kfold_cv(X, y, n_splits=5, random_state=42):
    """Perform Stratified K-Fold cross-validation"""
    print("\n" + "="*80)
    print("STRATIFIED K-FOLD CROSS-VALIDATION")
    print("="*80)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr_classifier = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )

    # Use StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Track metrics for each fold
    fold_results = []

    print(f"\nPerforming {n_splits}-fold stratified cross-validation...")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
        # Split data
        X_train_fold, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        # Train model
        lr_classifier.fit(X_train_fold, y_train_fold)

        # Predict
        y_pred_fold = lr_classifier.predict(X_test_fold)

        # Calculate metrics
        accuracy = accuracy_score(y_test_fold, y_pred_fold)
        precision = precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)
        recall = recall_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)
        f1 = f1_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)

        fold_results.append({
            'Fold': fold,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

        print(f"Fold {fold}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    # Create results dataframe
    results_df = pd.DataFrame(fold_results)

    print("\n" + "-"*80)
    print("CROSS-VALIDATION SUMMARY:")
    print("-"*80)
    print(f"Mean Accuracy:  {results_df['Accuracy'].mean():.4f}  {results_df['Accuracy'].std():.4f}")
    print(f"Mean Precision: {results_df['Precision'].mean():.4f}  {results_df['Precision'].std():.4f}")
    print(f"Mean Recall:    {results_df['Recall'].mean():.4f}  {results_df['Recall'].std():.4f}")
    print(f"Mean F1-Score:  {results_df['F1-Score'].mean():.4f}  {results_df['F1-Score'].std():.4f}")

    # Plot fold performance
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        ax.plot(results_df['Fold'], results_df[metric], marker='o', linewidth=2, markersize=8)
        ax.axhline(y=results_df[metric].mean(), color='r', linestyle='--',
                   label=f'Mean: {results_df[metric].mean():.4f}')
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Across Folds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(results_df['Fold'])

    plt.tight_layout()
    plt.savefig('kfold_performance_lr.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results_df


def kfold_cv_detailed(X, y, n_splits=5, random_state=42):
    """Detailed K-Fold cross-validation with multiple metrics"""
    print("\n" + "="*80)
    print("DETAILED K-FOLD CROSS-VALIDATION (Multiple Metrics)")
    print("="*80)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr_classifier = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )

    # Use StratifiedKFold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Define multiple scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision_weighted': 'precision_weighted',
        'recall_weighted': 'recall_weighted',
        'f1_weighted': 'f1_weighted'
    }

    # Perform cross-validation with multiple metrics
    cv_results = cross_validate(
        lr_classifier, X_scaled, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    # Display results
    print(f"\n{n_splits}-Fold Cross-Validation Results:")
    print("-"*80)

    for metric in scoring.keys():
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']

        print(f"\n{metric.upper()}:")
        print(f"  Train: {train_scores.mean():.4f} � {train_scores.std():.4f}")
        print(f"  Test:  {test_scores.mean():.4f} � {test_scores.std():.4f}")
        print(f"  Test scores per fold: {test_scores}")

    return cv_results


def hyperparameter_tuning(X, y, random_state=42):
    """Perform hyperparameter tuning using GridSearchCV"""
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING")
    print("="*80)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l2'],  # L2 regularization
        'solver': ['lbfgs', 'saga'],
        'max_iter': [500, 1000, 2000]
    }

    # Initialize base model
    lr_base = LogisticRegression(
        multi_class='multinomial',
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )

    # Perform grid search
    print("\nSearching for best parameters...")
    print("This may take a while...")

    grid_search = GridSearchCV(
        lr_base,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_scaled, y)

    print("\n" + "="*80)
    print("BEST PARAMETERS FOUND")
    print("="*80)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Show top 5 parameter combinations
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_results = results_df.nlargest(5, 'mean_test_score')[
        ['params', 'mean_test_score', 'std_test_score']
    ]

    print("\nTop 5 Parameter Combinations:")
    for idx, row in top_results.iterrows():
        print(f"\n{row['params']}")
        print(f"  Score: {row['mean_test_score']:.4f} � {row['std_test_score']:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def main():
    """Main function to run the complete Logistic Regression analysis"""
    # Prepare data
    X, y, categorical_features = prepare_data()

    # Train Logistic Regression
    lr_classifier, scaler, X_train, X_test, y_train, y_test, y_pred_test, y_pred_proba = train_logistic_regression(X, y)

    # Evaluate model
    evaluate_model(lr_classifier, X_test, y_test, y_pred_test)

    # Feature coefficients analysis
    feature_coefficients(lr_classifier, categorical_features)

    # K-Fold Cross-Validation Techniques
    print("\n" + "="*80)
    print("APPLYING K-FOLD CROSS-VALIDATION TECHNIQUES")
    print("="*80)

    # 1. Stratified K-Fold
    stratified_kfold_cv(X, y, n_splits=5)

    # 2. Detailed cross-validation
    kfold_cv_detailed(X, y, n_splits=5)

    # Uncomment below to perform hyperparameter tuning (takes longer)
    # best_model, best_params = hyperparameter_tuning(X, y)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - confusion_matrix_lr.png")
    print("  - feature_coefficients_lr.png")
    print("  - kfold_performance_lr.png")


if __name__ == "__main__":
    main()
