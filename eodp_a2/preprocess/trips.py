import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import json

def process_trips(filepath):
    df = pd.read_csv(filepath, on_bad_lines='skip', low_memory=False)

    df['transport_mode'] = df['linkmode'].apply(categorize_mode)
    #clean nan values from trips dataframe
    df = df.dropna(subset=['transport_mode'])

    '''
    # Plotting the distribution of transport categories
    plt.figure(figsize=(10, 6))
    sns.countplot(data=trips, x='transport_mode', order=trips['transport_mode'].value_counts().index, palette='Set2')
    plt.title('Distribution of Transport Categories')
    plt.xlabel('Transport Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    '''

    # Define the columns to extract
    columns_to_extract = [
        'tripid', 'hhid', 'persid', 'tripno', 'trippoststratweight', 
        'transport_mode', 'trippurp', 'starthour','arrhour','travtime',
        'cumdist',  'origlga', 'destpurp1',
        'destlga', 'dayType', 'homesubregion_ASGS', 'homeregion_ASGS', 
    ]
    # Extract the specified columns and save to a CSV file
    prefer_trips = df[columns_to_extract]

    return prefer_trips

def categorize_mode(mode):
    if mode in ['Train', 'Public Bus', 'Tram', 'School Bus']:
        return 'Public'
    elif mode in ['Vehicle Driver', 'Vehicle Passenger', 'Motorcycle', 'Taxi','Rideshare Service']:
        return 'Private'
    elif mode in ['Walking', 'Bicycle', 'Running/jogging', 'e-Scooter']:
        return 'Active'
    else:
        return 'Other'
