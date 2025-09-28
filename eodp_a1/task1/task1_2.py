"""
I Acknowledge the use of Chatgpt to help me make graphs 
I entered the followering prompt "How do I make a heatmap and piechart using python data analysis"
I used the output to find out what libraries and functions will make a heatmap and a pie chart
"""
from task1_1 import task1_1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def task1_2():
    df= pd.read_csv("task1.csv")
    # crosstabs makes a tabled relationship of aveagegroup_5 and hhinc_group
    heatmap_data = pd.crosstab(df['aveagegroup_5'], df['hhinc_group'])
    print(heatmap_data.head)
    print(df.describe)
    plt.figure(figsize=(10,6))
    sns.heatmap(heatmap_data,cmap="Blues")
    plt.title("Relationship between Age group and Household income")
    plt.ylabel("Age Group (Decades)")
    plt.xlabel("Household Income")

    plt.savefig("task1_heatmap.png")
    plt.close()
    
    # Makes seperate series/columns for city and shire
    cities = df[df['homelga'].str.contains("(C)")]
    shires = df[df['homelga'].str.contains("(S)")]
    
    # counts the specific values found inside the series'
    city_counts= cities['hhinc_group'].value_counts()
    shire_counts=shires['hhinc_group'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(16,8))
    print(city_counts.head)
    axes[0].pie(city_counts, labels = city_counts.index, autopct= '%1.1f%%')
    axes[0].set_title("Income Distribution (Cities)")

    axes[1].pie(shire_counts, labels = shire_counts.index, autopct= '%1.1f%%')
    axes[1].set_title("Income Distribution (Shire)")

    plt.tight_layout()
    plt.savefig("task1_pie.png")
    plt.close()

    return
