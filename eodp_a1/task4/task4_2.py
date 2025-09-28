import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to group start times (in minutes after midnight) into categories
def time_group(m):
    if pd.isna(m):
        return None
    h = int(m)

    if h < 240:
        return "Before 4am"
    elif h < 300:
        return "4-5am"
    elif h < 420:
        return "5-7am"
    elif h < 540:
        return "7-9am"
    else:
        return "After 9am"

def task4_2():
    # Load only the required columns to save memory
    usecols = ["dayType", "start_time", "journey_weight"]
    jte = pd.read_csv(r"/Users/phamthiphuongthuy/Desktop/Unimelb/Y2S1/EODP/eodp_a1/journey_to_education_vista_2023_2024.csv", usecols=usecols)
    jtw = pd.read_csv(r"/Users/phamthiphuongthuy/Desktop/Unimelb/Y2S1/EODP/eodp_a1/journey_to_work_vista_2023_2024.csv", usecols=usecols)

    # Add time group (based on start_time) and dataset labels
    for df, label in [(jte, "Education"), (jtw, "Work")]:
        df["TimeGroup"] = df["start_time"].apply(time_group)
        df["Dataset"] = label

    # Combine both datasets
    all_data = pd.concat([jte, jtw])

    # Aggregate weighted journey counts by day type, time group, and dataset
    counts = (
        all_data
        .groupby(["dayType", "TimeGroup", "Dataset"])["journey_weight"]
        .sum()
        .reset_index()
    )

    # --------- Plotting section (developed with AI assistance, ChatGPT) ---------
    # Two subplots: one for Weekday, one for Weekend
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    for ax, day in zip(axes, ["Weekday", "Weekend"]):
        sub = counts[counts["dayType"] == day]
        sns.barplot(
            data=sub, x="TimeGroup", y="journey_weight",
            hue="Dataset", ax=ax
        )
        ax.set_title(day)

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig("task4_2.png")
    plt.close()
