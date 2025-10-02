import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def visualisation():
    # Load the data
    data = pd.read_csv('../preprocess/trips_demographic.csv')
    
    # Create a count plot
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='owndwell', hue='transport_mode')
    
    # Add labels and title
    plt.xlabel('Household Income (persinc)')
    plt.ylabel('Count')
    plt.title('Count of Each Transport Mode vs Personal Income')
    plt.legend(title='Transport Mode')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

visualisation()