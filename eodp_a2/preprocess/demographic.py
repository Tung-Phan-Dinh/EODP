import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocess_person(inputfile):
    df = pd.read_csv(inputfile)
    
    wfh_days = ['wfhmon', 'wfhtue', 'wfhwed', 'wfhthu', 'wfhfri', 'wfhsat', 'wfhsun']
    wfh_mapping = {'Yes': 1, 'No': 0, 'Not in Work Force': 0}

    toll = ['anytoll', 'anyvehwalk', 'anypaidpark']
    toll_mapping = {'Yes': 1, 'No': 0, 'N/A - Did not use a car': 0}

    other = ['mbikelicence', 'otherlicence', 'fulltimework', 'parttimework', 'casualwork', 'anywork']
    other_mapping = {'Yes': 1, 'No': 0, 'Not applicable': 0}

    # Mode imputation for missing values
    for number in range(1, 11):
        mode_value = df[f"perspoststratweight_GROUP_{number}"].mode()
        if not mode_value.empty:  # Check if mode exists
            df[f"perspoststratweight_GROUP_{number}"] = df[f"perspoststratweight_GROUP_{number}"].fillna(mode_value[0])

    df = mapping(df, wfh_days, wfh_mapping)
    df = mapping(df, toll, toll_mapping)
    df = mapping(df, other, other_mapping)

    # Encode income ranges with start of range
    income_mapping = {
        'Nil income': 0,
        'Negative income': 0,
        '$1-$149 ($1-$7,799)': 1,
        '$150-$299 ($7,800-$15,599)': 150,
        '$300-$399 ($15,600-$20,799)': 300,
        '$400-$499 ($20,800-$25,999)': 400,
        '$500-$649 ($26,000-$33,799)': 500,
        '$650-$799 ($33,800-$41,599)': 650,
        '$800-$999 ($41,600-$51,999)': 800,
        '$1,000-$1,249 ($52,000-$64,999)': 1000,
        '$1,250-$1,499 ($65,000-$77,999)': 1250,
        '$1,500-$1,749 ($78,000-$90,999)': 1500,
        '$1,750-$1,999 ($91,000-$103,999)': 1750,
        '$2,000-$2,999 ($104,000-$155,999)': 2000,
        '$3,000-$3,499 ($156,000-$181,999)': 3000,
        '$3,500 or more ($182,000 or more)': 3500
    }
    df['persinc'] = df['persinc'].map(income_mapping)

    # Categorize person income into low/medium/high
    def categorize_person_income(income):
        if income <= 500:  # Up to $26k annually
            return 'low'
        elif income <= 1500:  # $26k to $78k annually
            return 'medium'
        else:  # Above $78k annually
            return 'high'

    df['persinc_category'] = df['persinc'].apply(categorize_person_income)

    # Add person income column
    df.drop(['persno', 'relationship', 'anzsco1', 'anzsco2', 'anzsic1',
             'anzsic2', 'faretype', 'anytoll', 'anyvehwalk', 'anypaidpark',
             'perspoststratweight', 'perspoststratweight_GROUP_1', 'perspoststratweight_GROUP_2',
             'perspoststratweight_GROUP_3', 'perspoststratweight_GROUP_4', 'perspoststratweight_GROUP_5',
             'perspoststratweight_GROUP_6', 'perspoststratweight_GROUP_7', 'perspoststratweight_GROUP_8',
             'perspoststratweight_GROUP_9', 'perspoststratweight_GROUP_10'], axis=1, inplace=True)
    return df

def preprocess_household(inputfile):
    df = pd.read_csv(inputfile)

    columns = ['youngestgroup_5', 'aveagegroup_5', 'oldestgroup_5']
    df['hhinc_group'] = df['hhinc_group'].fillna(df['hhinc_group'].mode()[0])

    # Encode household income ranges with start of range
    hh_income_mapping = {
        '$1-$149 ($1-$7,799)': 1,
        '$150-$299 ($7,800-$15,599)': 150,
        '$300-$399 ($15,600-$20,799)': 300,
        '$400-$499 ($20,800-$25,999)': 400,
        '$500-$649 ($26,000-$33,799)': 500,
        '$650-$799 ($33,800-$41,599)': 650,
        '$800-$999 ($41,600-$51,999)': 800,
        '$1,000-$1,249 ($52,000-$64,999)': 1000,
        '$1,250-$1,499 ($65,000-$77,999)': 1250,
        '$1,500-$1,749 ($78,000-$90,999)': 1500,
        '$1,750-$1,999 ($91,000-$103,999)': 1750,
        '$2,000-$2,499 ($104,000-$129,999)': 2000,
        '$2,500-$2,999 ($130,000-$155,999)': 2500,
        '$3,000-$3,499 ($156,000-$181,999)': 3000,
        '$3,500-$3,999 ($182,000-$207,999)': 3500,
        '$4,000-$4,499 ($208,000-$233,999)': 4000,
        '$4,500-$4,999 ($234,000-$259,999)': 4500,
        '$5,000-$5,999 ($260,000-$311,999)': 5000,
        '$6,000-$7,999 ($312,000-$415,999)': 6000,
        '$8,000 or more ($416,000 or more)': 8000
    }

    df['homelga'] = df['homelga'].apply(extract_location_type)
    df['hhinc_group'] = df['hhinc_group'].map(hh_income_mapping)
    df['hhinc_category'] = df['hhinc_group'].apply(categorize_household_income)

    return df[['hhid', 'hhsize', 'dwelltype', 'owndwell', 'travdow',
                'aveagegroup_5', 'hhinc_group', 'hhinc_category',"homelga"]]

# Function for mapping multiple columns in dataframe
def mapping(dataframe, list_of_data, binary_mapping):
    for data in list_of_data:
        dataframe[data] = dataframe[data].map(binary_mapping)
    return dataframe
    
# Categorize household income into low/medium/high
def categorize_household_income(income):
    if income <= 1500:  # Up to $78k annually
        return 'low'
    elif income <= 3500:  # $78k to $182k annually
        return 'medium'
    else:  # Above $182k annually
        return 'high'

# Map homelga to City (C) or Shire (S)
def extract_location_type(location):
    if '(C)' in location:
        return 'C'
    elif '(S)' in location:
        return 'S'
    else:
        return location  # Keep original if neither pattern found

