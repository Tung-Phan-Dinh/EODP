import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

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

def correlation_heatmap():
    # Load the data
    data = pd.read_csv('../preprocess/trips_demographic.csv')

    # Encode categorical variables as numeric
    data['transport_mode_encoded'] = pd.factorize(data['transport_mode'])[0]
    data['sex_encoded'] = pd.factorize(data['sex'])[0]
    data['carlicence_encoded'] = pd.factorize(data['carlicence'])[0]
    data['anywork_encoded'] = pd.factorize(data['anywork'])[0]
    data['studying_encoded'] = pd.factorize(data['studying'])[0]
    data['mainact_encoded'] = pd.factorize(data['mainact'])[0]
    data['dwelltype_encoded'] = pd.factorize(data['dwelltype'])[0]
    data['owndwell_encoded'] = pd.factorize(data['owndwell'])[0]
    data['hhinc_category_encoded'] = pd.factorize(data['hhinc_category'])[0]

    # Select encoded demographic features and transport mode
    cols_of_interest = ['hhsize', 'sex_encoded', 'carlicence_encoded',
                        'anywork_encoded', 'studying_encoded', 'mainact_encoded',
                        'dwelltype_encoded', 'owndwell_encoded', 'hhinc_category_encoded',
                        'transport_mode_encoded']

    # Create correlation data and drop any NaN values
    corr_data = data[cols_of_interest].dropna()

    # Calculate the correlation matrix
    corr_matrix = corr_data.corr()

    # Extract correlations with transport_mode
    transport_correlations = corr_matrix['transport_mode_encoded'].drop('transport_mode_encoded')

    print("\nCorrelation with transport mode (sorted by absolute value):")
    print(transport_correlations.abs().sort_values(ascending=False))

    # Plot the full heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, center=0,
                xticklabels=['hhsize', 'sex', 'carlicence', 'anywork', 'studying',
                            'mainact', 'dwelltype', 'owndwell', 'hhinc', 'transport_mode'],
                yticklabels=['hhsize', 'sex', 'carlicence', 'anywork', 'studying',
                            'mainact', 'dwelltype', 'owndwell', 'hhinc', 'transport_mode'])
    plt.title('Correlation Heatmap: Demographic Features vs Transport Mode')
    plt.tight_layout()
    plt.show()

correlation_heatmap()