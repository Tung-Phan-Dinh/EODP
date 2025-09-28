import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import json
from task3_1 import categorize_transport

def task3_2():
    # Load journey to work data
    journey_to_work = r"/Users/phamthiphuongthuy/Desktop/Unimelb/Y2S1/EODP/eodp_a1/journey_to_work_vista_2023_2024.csv"

    data = pd.read_csv(journey_to_work)
    df = pd.DataFrame(data)
    # Calculate wasted time as difference between elapsed and travel time
    df['wasted_time'] = df['journey_elapsed_time'] - df['journey_travel_time']

    # Categorize wasted time into groups
    bins = [0, 1, 5, 10, 30, float('inf')]
    labels = ['0', '1-5', '5-10','10-30', '30+']
    df['group_wasted_time'] = pd.cut(df['wasted_time'], bins=bins, labels=labels, right=False)

    # Categorize transport modes using imported function
    df['transport_category'] = df['main_journey_mode'].apply(categorize_transport)

    # Categorize travel distance into groups
    bins = [0, 10, 20, 40, float('inf')]
    dist_labels = ['0-10', '10-20', '20-40', '40+']
    df['group_travel_distance'] = pd.cut(df['journey_distance'], bins=bins, labels=dist_labels, right=False)
    df['group_travel_distance'].value_counts()

    percentage = {}
    # Count combinations of wasted time and distance groups
    dist_and_wasted_time = (
        df[['group_wasted_time','group_travel_distance']]
        .value_counts()
        .rename('count')         # turn Series values into a column name
        .reset_index()           # make the index regular columns
    )

    # Calculate percentage within each distance group
    dist_and_wasted_time['percentage'] = (
        dist_and_wasted_time.groupby("group_travel_distance", observed=False)["count"]
        .transform(lambda x: 100 * x / x.sum())
        )

    # Create nested dictionary structure for JSON output
    result = {}
    for dist_group in dist_labels:
        subset = dist_and_wasted_time[dist_and_wasted_time['group_travel_distance'].eq(dist_group)]
        result[dist_group] = {
            row['group_wasted_time']: {
                'count': int(row['count']),
                'percentage': round(row['percentage'], 2)
            }
            for _, row in subset.iterrows()
        }
    # Save results to JSON file
    with open('task3_2.json', 'w') as f:
        json.dump(result, f, indent=4)

        # Load the JSON data back for plotting
    with open('task3_2.json', 'r') as f:
        data = json.load(f)

    # Prepare data for plotting
    distance_groups = list(data.keys())

    # Set up the grouped bar chart
    plt.figure(figsize=(12, 8))
    x = np.arange(len(distance_groups))
    width = 0.15  # Width of bars


    # Create bars for each wasted time category
    for i, category in enumerate(labels):
        percentages = []
        # Get percentage for each distance group, or 0 if no data
        for dist_group in distance_groups:
            if category in data[dist_group]:
                percentages.append(data[dist_group][category]['percentage'])
            else:
                percentages.append(0)
        
        plt.bar(x + i * width, percentages, width, label=f'{category} min wasted', alpha=0.8)

    # Customize the plot
    plt.xlabel('Distance Group (km)')
    plt.ylabel('Percentage (%)')
    plt.title('Distribution of Wasted Time Categories by Distance Group')
    plt.xticks(x + width * 2, distance_groups)
    plt.legend(title='Wasted Time Groups')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig('task3_2.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return

"""
    Declaration
    I acknowledge the use of ChatGPT [https://chat.openai.com/] to support the
    development of my code and understanding of key concepts.

    I used prompts to:
    Understand and write Python functions for processing and aggregrating data.
    Generate visualisations with seaborn and matplotlib.
"""