import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import json

def process_trips(filepath):
    trips = pd.DataFrame(filepath)

    trips['transport_mode'] = trips['linkmode'].apply(categorize_mode)
    #clean nan values from trips dataframe
    trips = trips.dropna(subset=['transport_mode'])

    # Plotting the distribution of transport categories
    plt.figure(figsize=(10, 6))
    sns.countplot(data=trips, x='transport_mode', order=trips['transport_mode'].value_counts().index, palette='Set2')
    plt.title('Distribution of Transport Categories')
    plt.xlabel('Transport Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Define the columns to extract
    columns_to_extract = [
        'tripid', 'hhid', 'persid', 'tripno', 'trippoststratweight', 
        'transport_mode', 'trippurp', 'starthour', 'startime', 'arrhour', 
        'arrtime', 'travtime', 'triptime', 'cumdist',  'origlga',
        'destlga', 'dayType', 'homesubregion_ASGS', 'homeregion_ASGS', 
    ]
    # Extract the specified columns and save to a CSV file
    prefer_trips = trips[columns_to_extract]
    prefer_trips.to_csv("prefer_trips.csv", index=False)



def categorize_mode(mode):
    if mode in ['Train', 'Public Bus', 'Tram', 'School Bus', 'Plane']:
        return 'Public'
    elif mode in ['Vehicle Driver', 'Vehicle Passenger', 'Motorcycle', 'Taxi', 'Mobility Scooter']:
        return 'Private'
    elif mode in ['Walking', 'Bicycle', 'Running/jogging', 'e-Scooter']:
        return 'Active'
    else:
        return 'Other'
