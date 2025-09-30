import pandas as pd

def clean_stops(input_file, output_file):

    # Relevant columns
    cols = [
        "stopid", "hhid", "persid", "tripid", "stopno", "dayType",
        "starthour", "arrhour", "dephour", "startime", "arrtime", "deptime",
        "travtime", "vistadist", "duration",
        "mainmode", "fullmode", "origplace1", "destplace1",
        "origlga", "destlga", "stoppoststratweight",
        "homesubregion_ASGS", "homeregion_ASGS"
    ]

    # Read dataset
    df = pd.read_csv(input_file, usecols=lambda c: c in cols)


    # Convert numeric columns
    numeric_cols = ["starthour", "arrhour", "dephour", "startime", "arrtime", "deptime",
                    "travtime", "vistadist", "duration", "stoppoststratweight"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove impossible values
    for col in ["travtime", "vistadist", "duration", "stoppoststratweight"]:
        if col in df.columns:
            df = df[df[col] >= 0]

    # Handle missing values
    for col in ["travtime", "vistadist", "duration"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    if "stoppoststratweight" in df.columns:
        df["stoppoststratweight"] = df["stoppoststratweight"].fillna(0)
    for col in ["origplace1", "destplace1", "mainmode", "fullmode",
                "homesubregion_ASGS", "homeregion_ASGS"]:
        if col in df.columns:
            df[col] = df[col].fillna("Other")

    # Standardize text columns
    text_cols = ["origplace1", "destplace1", "mainmode", "fullmode",
                 "origlga", "destlga", "homesubregion_ASGS", "homeregion_ASGS"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # Drop duplicates
    df = df.drop_duplicates(subset="stopid")

    # Save cleaned dataset
    df.to_csv(output_file, index=False)


