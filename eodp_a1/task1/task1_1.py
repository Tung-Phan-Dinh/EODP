
import pandas as pd 
from lxml import etree

def task1_1():
    df= pd.read_csv(r"/Users/phamthiphuongthuy/Desktop/Unimelb/Y2S1/EODP/eodp_a1/household_vista_2023_2024 (1).csv")
    group=df['hhinc_group']
    # First thing is to get the average of the number in brackets

    ## 1) Removes number outside of brackets
    df['bracket_part'] = df['hhinc_group'].str.extract(r"\(([^)]+)\)") 

    # This makes it so that these cells average becomes 450000 later down
    df.loc[df['bracket_part']== "$416,000 or more", 'bracket_part']= "$416,000 - $484,000"

    # 2) Makes a copy and removes NaN to do calculations
    temp = df.dropna(subset=['bracket_part']).copy() ##removes NaN to do calculations
    print(temp['bracket_part'])

    # 3)Converts the two values to two columns of just numbers
    temp[['low', 'high']] = temp['bracket_part'].str.extract(r"\$?\s*([\d,]+)(?:\s*-\s*\$?\s*([\d,]+))?")

    # 4)Removes Commas
    temp[['low','high']] = (
    temp[['low','high']]
      .apply(lambda x: x.str.replace(',', '', regex=False))
    )   

    # 5) Makes cells into Int64 type for calculations
    temp[['low', 'high']] = temp[['low', 'high']].astype("Int64")
    temp['hhinc_group'] = temp[['low', 'high']].mean(axis=1)
    df['hhinc_group'] = temp['hhinc_group']
    df['hhinc_group'].round(2)
    
    ## Replace NaN with the mean figure
    mean_value=df['hhinc_group'].mean()
    mean_value.round(2)
    df['hhinc_group'] = df['hhinc_group'].fillna(mean_value)

    columns_to_process= df[['aveagegroup_5', 'oldestgroup_5', 'youngestgroup_5']]
    for col in columns_to_process:
        # splits the column into 'start' and 'end' via "->"
        part= df[col].str.split("->", n=1, expand=True)
        df['start']= pd.to_numeric(part[0], errors='coerce')
        df['end']= pd.to_numeric(part[1], errors='coerce')

        # makes mask so "0->9" only happens if start is just 5 not 50 or 55
        mask_5to9 = df['start'] == 5
        df.loc[mask_5to9, col] = "0->9"
        
    # changes the groups into categorisation by decades only (skips 0)
        for i in range(1,10):
            start= i * 10
            end=  start + 9
            df[col].astype('string')
            mask_decade = df[col].astype('string').str.startswith(str(i))
            df.loc[mask_decade, col] = f"{start}->{end}"

    # removes the start and end columns
    df.drop(columns=[['start','end']], inplace=True, errors="ignore")
    
    df.to_csv("task1.csv")

    
    return 
