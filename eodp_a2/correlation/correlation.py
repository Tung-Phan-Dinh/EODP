import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score

def visualisation():
    # Load the data
    data = pd.read_csv('../preprocess/trips_demographic.csv')
    
    # Calculate the proportion of each transport mode for each household size
    transport_counts = data.groupby(['hhsize', 'transport_mode']).size().unstack(fill_value=0)
    transport_props = transport_counts.div(transport_counts.sum(axis=1), axis=0)

    # Plot stacked bar chart
    transport_props.plot(
        kind='bar',
        stacked=True,
        figsize=(10, 6),
    )

    plt.xlabel('Household Size (hhsize)')
    plt.ylabel('Proportion')
    plt.title('Proportion of Each Transport Mode by Household Size')
    plt.legend(title='Transport Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

#visualisation()

def chi_square_test():
    """Perform chi-square test for relationship between categorical variables and transport mode"""
    # Load the data
    data = pd.read_csv('../preprocess/trips_demographic.csv')

    # Categorical features to test
    categorical_features = ['hhsize', 'sex', 'carlicence', 'anywork', 'studying',
                           'mainact', 'dwelltype', 'owndwell', 'hhinc_category', 'persinc_category']

    print("\n" + "="*80)
    print("CHI-SQUARE TEST RESULTS: Independence between features and transport mode")
    print("="*80)

    chi_square_results = []

    for feature in categorical_features:
        # Create contingency table
        contingency_table = pd.crosstab(data[feature], data['transport_mode'])

        # Perform chi-square test
        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        chi_square_results.append({
            'Feature': feature,
            'Chi-square': chi2,
            'p-value': p_value,
            'Significant (p<0.05)': 'Yes' if p_value < 0.05 else 'No'
        })

        print(f"\n{feature}:")
        print(f"  Chi-square statistic: {chi2:.4f}")
        print(f"  p-value: {p_value:.4e}")
        print(f"  Significant: {'Yes (p < 0.05)' if p_value < 0.05 else 'No (p >= 0.05)'}")

    # Create summary dataframe
    results_df = pd.DataFrame(chi_square_results)
    results_df = results_df.sort_values('Chi-square', ascending=False)

    print("\n" + "="*80)
    print("SUMMARY (sorted by Chi-square statistic):")
    print("="*80)
    print(results_df.to_string(index=False))

    return results_df


def mutual_information_analysis():
    """Calculate mutual information between features and transport mode"""
    # Load the data
    data = pd.read_csv('../preprocess/trips_demographic.csv')

    # Encode categorical variables
    categorical_features = ['hhsize', 'sex', 'carlicence', 'anywork', 'studying',
                           'mainact', 'dwelltype', 'owndwell', 'hhinc_category', 'persinc_category']

    # Encode all categorical variables including transport_mode
    encoded_data = data.copy()
    for col in categorical_features + ['transport_mode']:
        encoded_data[col + '_encoded'] = pd.factorize(encoded_data[col])[0]

    # Drop rows with NaN values
    encoded_data = encoded_data.dropna(subset=[f + '_encoded' for f in categorical_features] + ['transport_mode_encoded'])

    print("\n" + "="*80)
    print("MUTUAL INFORMATION ANALYSIS: Information gain from features about transport mode")
    print("="*80)

    mi_scores = []

    for feature in categorical_features:
        # Calculate mutual information
        mi_score = mutual_info_score(
            encoded_data[feature + '_encoded'],
            encoded_data['transport_mode_encoded']
        )

        mi_scores.append({
            'Feature': feature,
            'Mutual Information': mi_score
        })

        print(f"{feature}: {mi_score:.4f}")

    # Create summary dataframe
    mi_df = pd.DataFrame(mi_scores)
    mi_df = mi_df.sort_values('Mutual Information', ascending=False)

    print("\n" + "="*80)
    print("SUMMARY (sorted by Mutual Information):")
    print("="*80)
    print(mi_df.to_string(index=False))

    # Visualize mutual information scores
    plt.figure(figsize=(10, 6))
    plt.barh(mi_df['Feature'], mi_df['Mutual Information'])
    plt.xlabel('Mutual Information Score')
    plt.ylabel('Feature')
    plt.title('Mutual Information: Features vs Transport Mode')
    plt.tight_layout()
    plt.savefig('mutual_information.png')
    plt.show()

    return mi_df


def correlation_heatmap():
    """Create correlation heatmap using only numeric/ordinal variables"""
    # Load the data
    data = pd.read_csv('../preprocess/trips_demographic.csv')

    # Encode binary categorical variables (these have meaningful numeric interpretation)
    data['sex_encoded'] = pd.factorize(data['sex'])[0]
    data['carlicence_encoded'] = pd.factorize(data['carlicence'])[0]
    data['anywork_encoded'] = pd.factorize(data['anywork'])[0]
    data['studying_encoded'] = pd.factorize(data['studying'])[0]
    data['mainact_encoded'] = pd.factorize(data['mainact'])[0]
    data['dwelltype_encoded'] = pd.factorize(data['dwelltype'])[0]
    data['owndwell_encoded'] = pd.factorize(data['owndwell'])[0]
    data['hhinc_category_encoded'] = pd.factorize(data['hhinc_category'])[0]

    # One-hot encode transport_mode (creates binary columns for each mode)
    transport_dummies = pd.get_dummies(data['transport_mode'], prefix='transport')

    # Combine with other features
    analysis_data = pd.concat([
        data[['hhsize', 'sex_encoded', 'carlicence_encoded',
              'anywork_encoded', 'studying_encoded', 'mainact_encoded',
              'dwelltype_encoded', 'owndwell_encoded', 'hhinc_category_encoded']],
        transport_dummies
    ], axis=1)

    # Drop NaN values
    analysis_data = analysis_data.dropna()

    # Calculate the correlation matrix
    corr_matrix = analysis_data.corr()

    # Plot the full heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap: Demographic Features vs Transport Modes (One-Hot Encoded)')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Extract correlations with transport modes
    transport_cols = [col for col in corr_matrix.columns if col.startswith('transport_')]
    demographic_cols = [col for col in corr_matrix.columns if not col.startswith('transport_')]

    transport_corr = corr_matrix.loc[demographic_cols, transport_cols]

    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: Demographic features vs Transport modes")
    print("="*80)
    print("\nTop correlations with each transport mode:")
    for transport_col in transport_cols:
        print(f"\n{transport_col}:")
        top_corr = transport_corr[transport_col].abs().sort_values(ascending=False).head(5)
        for feature in top_corr.index:
            actual_corr = transport_corr.loc[feature, transport_col]
            print(f"  {feature}: {actual_corr:.4f}")

    return corr_matrix


def pearson_correlation():
    """Calculate Pearson correlation between transport modes and demographic features"""
    from scipy.stats import pearsonr

    # Load the data
    data = pd.read_csv('../preprocess/trips_demographic.csv')

    # Encode categorical variables
    data['sex_encoded'] = pd.factorize(data['sex'])[0]
    data['carlicence_encoded'] = pd.factorize(data['carlicence'])[0]
    data['anywork_encoded'] = pd.factorize(data['anywork'])[0]
    data['studying_encoded'] = pd.factorize(data['studying'])[0]
    data['mainact_encoded'] = pd.factorize(data['mainact'])[0]
    data['dwelltype_encoded'] = pd.factorize(data['dwelltype'])[0]
    data['owndwell_encoded'] = pd.factorize(data['owndwell'])[0]
    data['hhinc_category_encoded'] = pd.factorize(data['hhinc_category'])[0]
    data['persinc_category_encoded'] = pd.factorize(data['persinc_category'])[0]

    # One-hot encode transport_mode
    transport_dummies = pd.get_dummies(data['transport_mode'], prefix='transport')

    # Combine with demographic features
    analysis_data = pd.concat([
        data[['hhsize', 'sex_encoded', 'carlicence_encoded',
              'anywork_encoded', 'studying_encoded', 'mainact_encoded',
              'dwelltype_encoded', 'owndwell_encoded', 'hhinc_category_encoded',
              'persinc_category_encoded']],
        transport_dummies
    ], axis=1)

    # Drop NaN values
    analysis_data = analysis_data.dropna()

    print("\n" + "="*80)
    print("PEARSON CORRELATION: Transport modes vs Demographic features")
    print("="*80)

    # Get demographic and transport columns
    demographic_features = ['hhsize', 'sex_encoded', 'carlicence_encoded',
                           'anywork_encoded', 'studying_encoded', 'mainact_encoded',
                           'dwelltype_encoded', 'owndwell_encoded', 'hhinc_category_encoded',
                           'persinc_category_encoded']

    transport_modes = [col for col in analysis_data.columns if col.startswith('transport_')]

    # Store results
    pearson_results = []

    for transport_mode in transport_modes:
        print(f"\n{transport_mode}:")
        mode_results = []

        for feature in demographic_features:
            # Calculate Pearson correlation and p-value
            corr_coef, p_value = pearsonr(analysis_data[feature], analysis_data[transport_mode])

            mode_results.append({
                'Transport Mode': transport_mode,
                'Feature': feature,
                'Pearson r': corr_coef,
                'p-value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })

            print(f"  {feature:30s}: r={corr_coef:7.4f}, p={p_value:.4e} {'*' if p_value < 0.05 else ''}")

        # Sort by absolute correlation for this mode
        mode_results_sorted = sorted(mode_results, key=lambda x: abs(x['Pearson r']), reverse=True)
        pearson_results.extend(mode_results_sorted)

    # Create summary dataframe
    pearson_df = pd.DataFrame(pearson_results)

    print("\n" + "="*80)
    print("TOP CORRELATIONS (sorted by absolute Pearson r):")
    print("="*80)

    # Show top 15 strongest correlations overall
    top_correlations = pearson_df.reindex(
        pearson_df['Pearson r'].abs().sort_values(ascending=False).index
    ).head(15)

    print(top_correlations[['Transport Mode', 'Feature', 'Pearson r', 'p-value', 'Significant']].to_string(index=False))

    # Create visualization: heatmap of Pearson correlations
    pearson_matrix = analysis_data[demographic_features + transport_modes].corr().loc[
        demographic_features, transport_modes
    ]

    plt.figure(figsize=(10, 8))
    sns.heatmap(pearson_matrix, annot=True, fmt=".3f", cmap='coolwarm',
                vmin=-0.5, vmax=0.5, center=0, cbar_kws={'label': 'Pearson r'})
    plt.title('Pearson Correlation: Demographic Features vs Transport Modes')
    plt.xlabel('Transport Modes')
    plt.ylabel('Demographic Features')
    plt.tight_layout()
    plt.savefig('pearson_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    return pearson_df


# Run all analyses
chi_square_test()
mutual_information_analysis()
correlation_heatmap()
pearson_correlation()