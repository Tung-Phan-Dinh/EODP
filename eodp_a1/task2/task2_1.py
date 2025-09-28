import pandas as pd
import matplotlib.pyplot as plt

def task2_1():
    dataset = pd.read_csv("/course/person.csv")
    working_dataset = dataset[['wfhmon', 'wfhtue', 'wfhwed', 'wfhthu', 'wfhfri', 'wfhsat', 'wfhsun']]

    working_dataset = working_dataset.replace({'Yes' : 1, 'No' : 0, 'Not in Work Force' : -1})
    working_dataset['sum_of_days'] = working_dataset[['wfhmon', 'wfhtue', 'wfhwed', 'wfhthu', 'wfhfri', 'wfhsat', 'wfhsun']].sum(axis = 1)
    
    #Assisted by GenAI 
    #------------------
    working_dataset['wfh_category'] = pd.cut(working_dataset['sum_of_days'],
                                                bins = [0, 1, 3, 6, 8],
                                                labels = ['Never', 'Occasional', 'Frequent', 'Always'],
                                                right = False
    )
    #----------------------------------------------

    dataset = pd.concat([dataset, working_dataset['wfh_category']], axis = 1)
    dataset = dataset.dropna()

    group_total = dataset.groupby(['sex', 'emptype']).size().reset_index(name = 'group_total')
    category_total = dataset.groupby(['sex', 'emptype', 'wfh_category']).size().reset_index(name = 'category_total')
    
    proportions_df = pd.merge(category_total, group_total, on=['sex', 'emptype'])
    proportions_df['proportions_percentage'] = (proportions_df['category_total'] / proportions_df['group_total']) * 100

    # Assisted by GenAI 
    #-------------------
    proportions_df['combination'] = proportions_df['sex'] + ' + ' + proportions_df['emptype']
    
    pivot_data = proportions_df.pivot_table(
    index='combination',
    columns='wfh_category',
    values='proportions_percentage',
    aggfunc='first',
    fill_value=0
    )

    pivot_data = pivot_data.sort_index()

    # Create the stacked bar chart
    plt.figure(figsize=(14, 8))
    ax = pivot_data.plot(kind='bar', 
                    stacked=True,
                    figsize=(14, 8),
                    edgecolor='black',
                    linewidth=0.5,
                    colormap='viridis')  # You can choose different colormaps


    # Customize the plot
    plt.title('WFH Patterns by Gender and Employment Type', 
          fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Gender + Employment Type Combination', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("task2_1.png")
    #----------------------------
'''
Declaration

I acknowledge the use of Deepseek AI [deepseek.com] in helping me fine tune the visualisation, syntax errors and regarding how to use certain functions

I used the following prompts: "how to categorise data based on a column"
                                "For each combination of gender and employment type, should create a bar chart 
                                that is divided into multiple parts.

I used the output to better understand the syntax of the code and how the dataframe is manipulated              
'''
