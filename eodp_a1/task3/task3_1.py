import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


def task3_1():
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

            
    # Count number of stops from destpurp1_desc_01 to _15
    stop_labels = ['0 stop', '1 stop', '2 stops', '3+ stops']
    stop_cols = [f'destpurp1_desc_{str(i).zfill(2)}' for i in range(1, 16)]
    df['num_stops'] = df[stop_cols].notna().sum(axis=1) - 1  # Subtract 1 because destination is not a stop
    df['stop_category'] = df['num_stops'].apply(
        lambda x: f'{int(x)} stop' if x in [0, 1] else ('2 stops' if x == 2 else '3+ stops')
    )

    # Categorize transport modes using helper function
    df['transport_category'] = df['main_journey_mode'].apply(categorize_transport)
    # Remove rows with missing data
    filtered_df = df.dropna(subset=['group_wasted_time', 'transport_category', 'stop_category'])

    # Aggregate journey weights by wasted time, transport, and stop categories
    plot_data = (
        filtered_df.groupby(['group_wasted_time', 'transport_category', 'stop_category'],observed=True)['journey_weight']
        .sum()
        .reset_index()
    )
    # Calculate totals for each transport and stop category combination
    plot_data['total'] = plot_data.groupby(['transport_category', 'stop_category'])['journey_weight'].transform('sum')
    # Calculate proportions within each group
    plot_data['proportion'] = plot_data['journey_weight'] / plot_data['total']

    # Plot the distribution of wasted time
    transport_categories = ['Public', 'Private', 'Active']

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Distribution of Wasted Time Proportions by Transport Category', fontsize=16)

    for i, transport_cat in enumerate(transport_categories):
        # Filter data for this transport category
        subset_data = plot_data[plot_data['transport_category'] == transport_cat]
        
        # Create the plot
        sns.barplot(data=subset_data, 
                    x='group_wasted_time', 
                    y='proportion',
                    hue='stop_category',
                    hue_order=stop_labels,
                    palette='viridis',
                    ax=axes[i],
                    order=labels)
        
        # Customize each subplot
        axes[i].set_title(f'{transport_cat} Transport', fontsize=14)
        axes[i].set_xlabel('Wasted Time (minutes)')
        axes[i].set_ylabel('Proportion')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Only show legend for the last subplot
        if i < 2:
            axes[i].get_legend().remove()
        else:
            axes[i].legend(title='Number of Stops', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('task3_1.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return 



def categorize_transport(t_mode):
    """Categorize transport modes into Public, Private, Active, or Other"""
    if t_mode in ['Train', 'Public Bus', 'Tram', 'School Bus']:
        return 'Public'
    elif t_mode in ['Vehicle Driver', 'Vehicle Passenger', 'Motorcycle', 'Taxi / Rideshare']:
        return 'Private'
    elif t_mode in ['Walking', 'Bicycle']:
        return 'Active'
    else:
        return 'Other'
    
"""
    Declaration
    I acknowledge the use of ChatGPT [https://chat.openai.com/] to support the
    development of my code and understanding of key concepts.

    I used prompts to:
    Understand and write Python functions for processing and aggregrating data.
    Generate visualisations with seaborn and matplotlib.
    """