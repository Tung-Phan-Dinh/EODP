import pandas as pd
import matplotlib.pyplot as plt

# Function to bin travel time into four categories
def bin_time(x):
    if x < 16:
        return "0-15"
    elif x < 31:
        return "15-30"
    elif x < 61:
        return "30-60"
    else:
        return "60+"

# Function to map home subregion into Inner / Middle / Outer / Other
def map_region(x):
    if pd.isna(x):  
        return "Other"
    x = str(x).lower() 
    if "inner" in x:
        return "Inner"
    elif "middle" in x:
        return "Middle"
    elif "outer" in x:
        return "Outer"
    else:
        return "Other"

def task4_1():
    # -----------------------------
    # Journey to Education
    # -----------------------------
    cols = ["dayType", "journey_travel_time", "homesubregion_ASGS", "journey_weight"]
    df = pd.read_csv(r"/Users/phamthiphuongthuy/Desktop/Unimelb/Y2S1/EODP/eodp_a1/journey_to_education_vista_2023_2024.csv", usecols=cols)

    # Keep only weekday records
    df = df[df["dayType"] == "Weekday"].copy()
    
    # Add region and travel time bin columns
    df["Region"] = df["homesubregion_ASGS"].apply(map_region)
    df["TT_Bin"] = df["journey_travel_time"].apply(bin_time)

    # Cross-tabulation: sum of journey weights by region and time bin
    jte_xtab = pd.crosstab(
        df["Region"], df["TT_Bin"], 
        values=df["journey_weight"], aggfunc="sum"
    )

    # --------- Plotting section (developed with AI assistance, ChatGPT) ---------
    ax = jte_xtab.plot(kind="bar", figsize=(8, 6))
    ax.set_title("Journey to Education - Travel Time by Region (Weekday)")
    ax.set_xlabel("Home Suburb Region")
    ax.set_ylabel("Number of Journeys")
    ax.legend(title="Travel Time (minutes)")
    plt.tight_layout()
    plt.savefig("task4_1_JTE.png")

    # -----------------------------
    # Journey to Work
    # -----------------------------
    jtw = pd.read_csv(r"/Users/phamthiphuongthuy/Desktop/Unimelb/Y2S1/EODP/eodp_a1/journey_to_work_vista_2023_2024.csv", usecols=cols)

    # Keep only weekday records
    jtw = jtw[jtw["dayType"] == "Weekday"].copy()

    # Add region and travel time bin columns
    jtw["Region"] = jtw["homesubregion_ASGS"].apply(map_region)
    jtw["TT_Bin"] = jtw["journey_travel_time"].apply(bin_time)

    # Cross-tabulation: sum of journey weights by region and time bin
    jtw_xtab = pd.crosstab(
        jtw["Region"], jtw["TT_Bin"], 
        values=jtw["journey_weight"], aggfunc="sum"
    )

    # --------- Plotting section (developed with AI assistance, ChatGPT) ---------
    ax = jtw_xtab.plot(kind="bar", figsize=(8, 6))
    ax.set_title("Journey to Work - Travel Time by Region (Weekday)")
    ax.set_xlabel("Home Suburb Region")
    ax.set_ylabel("Number of Journeys")
    ax.legend(title="Travel Time (minutes)")
    plt.tight_layout()
    plt.savefig("task4_1_JTW.png")
    plt.close()
