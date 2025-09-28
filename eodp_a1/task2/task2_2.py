import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def task2_2():
    person_dataset = pd.read_csv(r"/Users/phamthiphuongthuy/Desktop/Unimelb/Y2S1/EODP/eodp_a1/person_vista_2023_2024.csv")
    trip_dataset = pd.read_csv("/Users/phamthiphuongthuy/Desktop/Unimelb/Y2S1/EODP/eodp_a1/trips_vista_2023_2024 (1).csv")
    person_dataset = person_dataset[person_dataset['emptype'] != 'Not in Work Force']
    working_dataset = person_dataset[['wfhmon', 'wfhtue', 'wfhwed', 'wfhthu', 'wfhfri', 'wfhsat', 'wfhsun']]


    working_dataset = working_dataset.replace({'Yes' : 1, 'No' : 0})
    working_dataset['sum_of_days'] = working_dataset[['wfhmon', 'wfhtue', 'wfhwed', 'wfhthu', 'wfhfri', 'wfhsat', 'wfhsun']].sum(axis = 1)
    
    #Assisted by GenAI 
    #------------------
    
    working_dataset['wfh_category'] = pd.cut(working_dataset['sum_of_days'],
                                                bins = [0, 1, 3, 6, 8],
                                                labels = ['Never', 'Occasional', 'Frequent', 'Always'],
                                                right = False
    )
    #----------------------------------------------

    person_dataset = pd.concat([person_dataset, working_dataset], axis = 1)
    person_dataset = person_dataset.dropna(subset=['sex', 'emptype'])
    
    trips_total = trip_dataset.groupby(['persid']).size().reset_index(name = 'trips_total')
    person_dataset = person_dataset.merge(
        trips_total,
        on = 'persid',
        how = 'inner'
    )
    
    average_trips_df = person_dataset.groupby(['sex', 'emptype', 'wfh_category']).agg(
            group_size=('persid', 'count'),      # Number of people in this group
            total_trips=('trips_total', 'sum'),  # Sum of all trips made by this group
            average_trips=('trips_total', 'mean'), # Average trips per person
        ).reset_index()
    
    wfh_order = ['Never', 'Occasional', 'Frequent', 'Always']
    average_trips_df['wfh_category'] = pd.Categorical(average_trips_df['wfh_category'], 
                                                 categories=wfh_order, ordered=True)
    average_trips_df = average_trips_df.sort_values(['sex', 'emptype', 'wfh_category'])

    #average_trips_df = person_dataset.groupby(['wfh_category', 'sex', 'emptype'])['trips_total'].mean().reset_index()
    average_trips_df['group'] = average_trips_df['sex'] + ' - ' + average_trips_df['emptype']

    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("husl", n_colors=average_trips_df['group'].nunique())

    for idx, (label, subdf) in enumerate(average_trips_df.groupby('group')):
        plt.plot(subdf['wfh_category'], subdf['average_trips'], label=label, color=palette[idx])

    plt.title("Average Number of Trips by WFH Category for Each (Sex, Emptype) Group")
    plt.xlabel("WFH Frequency Category")
    plt.ylabel("Average Number of Trips")
    plt.legend(title="Group (Sex – Emptype)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("another_task2_2.png")

'''Declaration

I acknowledge the use of Deepseek AI [deepseek.com] in helping me fine tune the visualisation, syntax errors and regarding how to use certain functions

I used the following prompts: "how to categorise data based on a column"
                                "For each combination of gender and employment type, should create a bar chart 
                                that is divided into multiple parts."
                                "i have two data frames, they both have a persid column but one of the dataframes is processed. 
                                How can i use the column of the processed dataframe in my other dataframe"

I used the output to better understand the syntax of the code and how the dataframe is manipulated'''  


