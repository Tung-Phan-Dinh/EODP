import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV,
                                      KFold, StratifiedKFold, cross_validate)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_data():
    """Load and prepare data for Random Forest classification"""
    # Load the data
    data = pd.read_csv('../preprocess/trips_demographic.csv')

    # Select features based on correlation analysis
    # Categorical features to encode
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


def train_random_forest(X, y, test_size=0.2, random_state=42):
    """Train Random Forest classifier"""
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("="*80)
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Initialize Random Forest with reasonable default parameters
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )

    # Train the model
    print("\nTraining model...")
    rf_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred_train = rf_classifier.predict(X_train)
    y_pred_test = rf_classifier.predict(X_test)

    # Evaluate performance
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return rf_classifier, X_train, X_test, y_train, y_test, y_pred_test


def evaluate_model(rf_classifier, X_test, y_test, y_pred_test):
    """Evaluate the Random Forest model"""
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
                xticklabels=rf_classifier.classes_,
                yticklabels=rf_classifier.classes_)
    plt.title('Confusion Matrix - Random Forest Classifier')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
    plt.show()

    return cm


def feature_importance_analysis(rf_classifier, feature_names):
    """Analyze and visualize feature importance"""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    # Get feature importances
    importances = rf_classifier.feature_importances_

    # Create dataframe for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importances (sorted):")
    print(feature_importance_df.to_string(index=False))

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance in Random Forest Model')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_rf.png', dpi=300, bbox_inches='tight')
    plt.show()

    return feature_importance_df


def stratified_kfold_cv(X, y, n_splits=5, random_state=42):
    """Perform Stratified K-Fold cross-validation (maintains class distribution in each fold)"""
    print("\n" + "="*80)
    print("STRATIFIED K-FOLD CROSS-VALIDATION")
    print("="*80)

    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )

    # Use StratifiedKFold to maintain class proportions in each fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Track metrics for each fold
    fold_results = []
    all_y_true = []
    all_y_pred = []

    print(f"\nPerforming {n_splits}-fold stratified cross-validation...")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        # Split data
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        # Train model
        rf_classifier.fit(X_train_fold, y_train_fold)

        # Predict
        y_pred_fold = rf_classifier.predict(X_test_fold)

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

        # Store predictions for overall confusion matrix
        all_y_true.extend(y_test_fold)
        all_y_pred.extend(y_pred_fold)

        print(f"Fold {fold}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    # Create results dataframe
    results_df = pd.DataFrame(fold_results)

    print("\n" + "-"*80)
    print("CROSS-VALIDATION SUMMARY:")
    print("-"*80)
    print(f"Mean Accuracy:  {results_df['Accuracy'].mean():.4f} ± {results_df['Accuracy'].std():.4f}")
    print(f"Mean Precision: {results_df['Precision'].mean():.4f} ± {results_df['Precision'].std():.4f}")
    print(f"Mean Recall:    {results_df['Recall'].mean():.4f} ± {results_df['Recall'].std():.4f}")
    print(f"Mean F1-Score:  {results_df['F1-Score'].mean():.4f} ± {results_df['F1-Score'].std():.4f}")

    # Plot fold performance
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        ax.plot(results_df['Fold'], results_df[metric], marker='o', linewidth=2, markersize=8)
        ax.axhline(y=results_df[metric].mean(), color='r', linestyle='--', label=f'Mean: {results_df[metric].mean():.4f}')
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Across Folds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(results_df['Fold'])

    plt.tight_layout()
    plt.savefig('kfold_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Overall confusion matrix from all folds
    cm_overall = confusion_matrix(all_y_true, all_y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y),
                yticklabels=np.unique(y))
    plt.title('Overall Confusion Matrix (All Folds Combined)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix_kfold.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results_df, all_y_true, all_y_pred


def kfold_cv_detailed(X, y, n_splits=5, random_state=42):
    """Detailed K-Fold cross-validation with multiple metrics"""
    print("\n" + "="*80)
    print("DETAILED K-FOLD CROSS-VALIDATION (Multiple Metrics)")
    print("="*80)

    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )

    # Use StratifiedKFold for classification
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
        rf_classifier, X, y,
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
        print(f"  Train: {train_scores.mean():.4f} ± {train_scores.std():.4f}")
        print(f"  Test:  {test_scores.mean():.4f} ± {test_scores.std():.4f}")
        print(f"  Test scores per fold: {test_scores}")

    return cv_results


def compare_fold_strategies(X, y, random_state=42):
    """Compare different folding strategies"""
    print("\n" + "="*80)
    print("COMPARING DIFFERENT FOLDING STRATEGIES")
    print("="*80)

    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )

    strategies = {
        'KFold (5-fold)': KFold(n_splits=5, shuffle=True, random_state=random_state),
        'StratifiedKFold (5-fold)': StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        'KFold (10-fold)': KFold(n_splits=10, shuffle=True, random_state=random_state),
        'StratifiedKFold (10-fold)': StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state),
    }

    comparison_results = []

    for strategy_name, cv_strategy in strategies.items():
        scores = cross_val_score(rf_classifier, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=-1)

        comparison_results.append({
            'Strategy': strategy_name,
            'Mean Accuracy': scores.mean(),
            'Std Dev': scores.std(),
            'Min': scores.min(),
            'Max': scores.max()
        })

        print(f"\n{strategy_name}:")
        print(f"  Mean Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"  Range: [{scores.min():.4f}, {scores.max():.4f}]")

    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_results)

    print("\n" + "-"*80)
    print("COMPARISON SUMMARY:")
    print("-"*80)
    print(comparison_df.to_string(index=False))

    # Visualize comparison
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(comparison_df))
    plt.bar(x_pos, comparison_df['Mean Accuracy'], yerr=comparison_df['Std Dev'],
            capsize=5, alpha=0.7, color='steelblue')
    plt.xlabel('Folding Strategy')
    plt.ylabel('Mean Accuracy')
    plt.title('Comparison of Different Cross-Validation Strategies')
    plt.xticks(x_pos, comparison_df['Strategy'], rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('cv_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return comparison_df


def hyperparameter_tuning(X, y, random_state=42):
    """Perform hyperparameter tuning using GridSearchCV"""
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING")
    print("="*80)

    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [10, 20, 50],
        'min_samples_leaf': [5, 10, 20]
    }

    # Initialize base model
    rf_base = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )

    # Perform grid search
    print("\nSearching for best parameters...")
    print("This may take a while...")

    grid_search = GridSearchCV(
        rf_base,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    print("\n" + "="*80)
    print("BEST PARAMETERS FOUND")
    print("="*80)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def main():
    """Main function to run the complete Random Forest analysis"""
    # Prepare data
    X, y, categorical_features = prepare_data()

    # Train Random Forest (simple train-test split)
    rf_classifier, X_train, X_test, y_train, y_test, y_pred_test = train_random_forest(X, y)

    # Evaluate model
    evaluate_model(rf_classifier, X_test, y_test, y_pred_test)

    # Feature importance analysis
    feature_importance_analysis(rf_classifier, categorical_features)

    # K-Fold Cross-Validation Techniques
    print("\n" + "="*80)
    print("APPLYING K-FOLD CROSS-VALIDATION TECHNIQUES")
    print("="*80)

    # 1. Stratified K-Fold (recommended for imbalanced classes)
    stratified_kfold_cv(X, y, n_splits=5)

    # 2. Detailed cross-validation with multiple metrics
    kfold_cv_detailed(X, y, n_splits=5)

    # 3. Compare different folding strategies
    compare_fold_strategies(X, y)

    # Uncomment below to perform hyperparameter tuning (takes longer)
    # best_model, best_params = hyperparameter_tuning(X, y)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - confusion_matrix_rf.png")
    print("  - feature_importance_rf.png")
    print("  - kfold_performance.png")
    print("  - confusion_matrix_kfold.png")
    print("  - cv_strategy_comparison.png")


if __name__ == "__main__":
    main()
