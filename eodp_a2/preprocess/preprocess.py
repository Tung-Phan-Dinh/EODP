import pandas as pd 
from jtwAndjte import clean_journey
from demographic import preprocess_household, preprocess_person
from trips import process_trips

edu = "../A2 datasets/journey_to_education_vista_2023_2024.csv"
work ="../A2 datasets/journey_to_work_vista_2023_2024.csv"
person = "../A2 datasets/person_vista_2023_2024.csv"
household =  "../A2 datasets/household_vista_2023_2024.csv"
trips = "../A2 datasets/trips_vista_2023_2024.csv"

def trip_comparison():
    edu_data = clean_journey(edu)
    work_data = clean_journey(work)
    person_data = preprocess_person(person)
    household_data = preprocess_household(household)

    # Merge household data onto person data using 'hhid'
    demographic = person_data.merge(household_data, on='hhid', suffixes=('', '_household'))
    # Remove duplicate columns if any
    demographic = demographic.loc[:, ~demographic.columns.duplicated()]

    # Add a label column to distinguish work and edu
    work_data['journey_label'] = 1
    edu_data['journey_label'] = 0
    # Combine work and edu into a single DataFrame
    journey = pd.concat([work_data, edu_data], ignore_index=True)

    # Merge demographic data onto journey data using 'persid' and 'hhid'
    merged_data = journey.merge(demographic, on=['persid', 'hhid'], suffixes=('_journey', '_demographic'))
    merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]
    merged_data.drop(["homesubregion_ASGS_demographic","homeregion_ASGS_demographic","dayType_demographic", "travdow_household"],axis=1, inplace=True)
    drop_cols = [
        'persid','hhid','travdow','otherlicence','nolicence','fulltimework','parttimework','casualwork',
        'activities','emptype','startplace','numstops','wfhmon','wfhtue','wfhwed','wfhthu','wfhfri',
        'wfhsat','wfhsun','wfhtravday','homesubregion_ASGS','dayType','travdow_household','aveagegroup_5',
        'homelga'
    ]
    merged_data = merged_data.drop(columns=[c for c in drop_cols if c in merged_data.columns])
    merged_data.to_csv("journey_compare.csv", index=False)

    return demographic

df = trip_comparison()

def demographic(df):
    trips_data = process_trips(trips)
    demographic = df

    # Merge demographic data onto trips data using 'persid' and 'hhid'
    merged_data = trips_data.merge(demographic, on=['persid', 'hhid'], suffixes=('_trips', '_demographic'))
    merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]
    merged_data.drop(["homesubregion_ASGS_demographic","homeregion_ASGS_demographic","dayType_demographic", "travdow_household"],axis=1, inplace=True)
    drop_cols = [
        'persid','hhid','travdow','otherlicence','nolicence','fulltimework','parttimework','casualwork',
        'activities','startplace','numstops','wfhmon','wfhtue','wfhwed','wfhthu','wfhfri',
        'wfhsat','wfhsun','wfhtravday','homesubregion_ASGS','dayType','travdow_household','aveagegroup_5'
    ]
    merged_data = merged_data.drop(columns=[c for c in drop_cols if c in merged_data.columns])
    merged_data.to_csv("trips_demographic.csv", index=False)

demographic(df)