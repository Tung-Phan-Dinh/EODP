import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

def visualisation_hhsize():
    # Load the data
    data = pd.read_csv('../preprocess/trips_demographic.csv')
    
    prop_table = pd.crosstab(data['hhsize'], data['transport_mode'], normalize='index')
    
    plt.subplot(2, 2, 4)
    prop_table.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Transport Mode Proportions by Household Group')
    plt.xlabel('Household Group')
    plt.ylabel('Proportion')
    plt.legend(title='Transport Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('transport_proportions_bars3.png', dpi=300, bbox_inches='tight')
    plt.show()
    
#visualisation_hhsize()

def visualisation_dwelltype():
    data = pd.read_csv('../preprocess/trips_demographic.csv')

    prop_table = pd.crosstab(data['dwelltype'], data['transport_mode'], normalize='index')
    
    plt.subplot(2, 2, 4)
    prop_table.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Transport Mode Proportions by Dwelling Type ')
    plt.xlabel('Dwelling Type ')
    plt.ylabel('Proportion')
    plt.legend(title='Transport Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('dwelltype_proportions_bars3.png', dpi=300, bbox_inches='tight')
    plt.show()

#visualisation_dwelltype()

def visualisation_carlicence():
    # Load the data
    df = pd.read_csv('../preprocess/trips_demographic.csv')
    
    transport_hhsize_weighted = df.groupby(['carlicence', 'transport_mode'])['trippoststratweight'].sum().unstack(fill_value=0)

    # Calculate percentage within each household size
    transport_hhsize_percentage = transport_hhsize_weighted.div(transport_hhsize_weighted.sum(axis=1), axis=0) * 100

    # Plot stacked bar chart
    plt.figure(figsize=(12, 6))
    transport_hhsize_percentage.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title("Weighted Distribution of Transport Modes by Car Licence (Stacked)", fontsize=16)
    plt.xlabel("Car Licence", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.legend(title='Transport Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('carlicence_proportions_bars3.png', dpi=300, bbox_inches='tight')
    plt.show()

visualisation_carlicence()
visualisation_dwelltype()
visualisation_hhsize()