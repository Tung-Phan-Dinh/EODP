import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def eda():
    trip = "../preprocess/trips_demographic.csv"
    df = pd.read_csv(trip)
    transport_mode_counts = df.groupby('transport_mode')['trippoststratweight'].sum()
    transport_mode_percentage = (transport_mode_counts / transport_mode_counts.sum()) * 100
    plt.figure(figsize=(10, 6))
    sns.barplot(x=transport_mode_percentage.index, y=transport_mode_percentage.values, palette="viridis")
    plt.title("Weighted Distribution of Transport Modes", fontsize=16)
    plt.xlabel("Transport Mode", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    # Weighted distribution of transport mode by household size
    # Group by both hhsize and transport_mode, sum the trip weights
    transport_hhsize_weighted = df.groupby(['hhsize', 'transport_mode'])['trippoststratweight'].sum().unstack(fill_value=0)

    # Calculate percentage within each household size
    transport_hhsize_percentage = transport_hhsize_weighted.div(transport_hhsize_weighted.sum(axis=1), axis=0) * 100

    # Plot stacked bar chart
    plt.figure(figsize=(12, 6))
    transport_hhsize_percentage.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title("Weighted Distribution of Transport Modes by Household Size (Stacked)", fontsize=16)
    plt.xlabel("Household Size", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.legend(title='Transport Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('transport_mode_by_hhsize_stacked.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Weighted distribution of transport mode by total bikes
    transport_bikes_weighted = df.groupby(['totalbikes', 'transport_mode'])['trippoststratweight'].sum().unstack(fill_value=0)
    transport_bikes_percentage = transport_bikes_weighted.div(transport_bikes_weighted.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(12, 6))
    transport_bikes_percentage.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title("Weighted Distribution of Transport Modes by Total Bikes in Household", fontsize=16)
    plt.xlabel("Total Bikes", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.legend(title='Transport Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('transport_mode_by_totalbikes.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Weighted distribution of transport mode by total vehicles
    transport_vehs_weighted = df.groupby(['totalvehs', 'transport_mode'])['trippoststratweight'].sum().unstack(fill_value=0)
    transport_vehs_percentage = transport_vehs_weighted.div(transport_vehs_weighted.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(12, 6))
    transport_vehs_percentage.plot(kind='bar', stacked=True, colormap='plasma')
    plt.title("Weighted Distribution of Transport Modes by Total Vehicles in Household", fontsize=16)
    plt.xlabel("Total Vehicles", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.legend(title='Transport Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('transport_mode_by_totalvehs.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Count of totalvehs for each income group
    vehs_income = df.groupby(['hhinc_category', 'totalvehs']).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 6))
    vehs_income.plot(kind='bar', colormap='viridis')
    plt.title("Count of Total Vehicles by Household Income Category", fontsize=16)
    plt.xlabel("Household Income Category", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(title='Total Vehicles', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('totalvehs_by_income.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Alternative: Stacked bar chart showing proportions
    vehs_income_percentage = vehs_income.div(vehs_income.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(12, 6))
    vehs_income_percentage.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title("Distribution of Total Vehicles by Household Income Category (%)", fontsize=16)
    plt.xlabel("Household Income Category", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.legend(title='Total Vehicles', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('totalvehs_by_income_percentage.png', dpi=300, bbox_inches='tight')
    plt.show()

eda()