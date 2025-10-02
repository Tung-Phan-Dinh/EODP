import pandas as pd

def clean_journey(input_file):
    # Columns required for analysis
    cols = [
        "hhid", "persid", "dayType",
        "main_journey_mode", "journey_travel_time",
        "journey_distance", "journey_elapsed_time",
        "journey_weight", "homesubregion_ASGS", "homeregion_ASGS",'wasted_time','num_stops','stop_category','group_travel_distance'
    ]

    # Read CSV with only relevant columns
    df = pd.read_csv(input_file)


    # Create stop category based on number of stops
    stop_cols = [f'destpurp1_desc_{str(i).zfill(2)}' for i in range(1, 16)]
    df['num_stops'] = df[stop_cols].notna().sum(axis=1) - 1  # Subtract 1 because destination is not a stop
    df['stop_category'] = df['num_stops'].apply(
        lambda x: f'{int(x)} stop' if x in [0, 1] else ('2 stops' if x == 2 else '3+ stops')
    )

    # Categorize travel distance into groups
    bins = [0, 10, 20, 40, float('inf')]
    dist_labels = ['0-10', '10-20', '20-40', '40+']
    df['group_travel_distance'] = pd.cut(df['journey_distance'], bins=bins, labels=dist_labels, right=False)
    df['group_travel_distance'].value_counts()

    df['wasted_time'] = df['journey_elapsed_time'] - df['journey_travel_time']

    #Remove impossible values (negative numbers)
    for col in ["journey_travel_time", "journey_distance", "journey_weight"]:
        if col in df.columns:
            df = df[df[col].astype(str) != ""]  # drop empty strings
            df[col] = pd.to_numeric(df[col], errors="coerce")  # convert to numeric
            df = df[df[col] >= 0]  # keep only non-negative

    # Handle missing values
    if "journey_travel_time" in df.columns:
        df["journey_travel_time"] = df["journey_travel_time"].fillna(df["journey_travel_time"].median())
    if "journey_distance" in df.columns:
        df["journey_distance"] = df["journey_distance"].fillna(df["journey_distance"].median())
    if "journey_weight" in df.columns:
        df["journey_weight"] = df["journey_weight"].fillna(0)
    if "homesubregion_ASGS" in df.columns:
        df["homesubregion_ASGS"] = df["homesubregion_ASGS"].fillna("Other")
    if "homeregion_ASGS" in df.columns:
        df["homeregion_ASGS"] = df["homeregion_ASGS"].fillna("Other")

    # Standardize text columns (lowercase, remove extra spaces)
    for col in ["homesubregion_ASGS", "homeregion_ASGS", "main_journey_mode"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Keep only relevant columns
    df = df[cols]
    
    return df

