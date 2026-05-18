import pandas as pd
import os

def get_data():
    url = 'HR_Analytics_Project/data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv'
    df = pd.read_csv(url)

    return df
    