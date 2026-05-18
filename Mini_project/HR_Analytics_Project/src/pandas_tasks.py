import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_data

df = get_data()

# Question 1
def load_and_inspect():
    print('='*50 + ' Question 1 ' + '='*50)
    print("1. HIỂN THỊ 5 DÒNG ĐẦU (head):")
    print(df.head()) 
    print("\n2. THÔNG TIN DỮ LIỆU (info):")
    print("\n3. THỐNG KÊ MÔ TẢ (describe):")
    print(df.describe().round(2))

# Question 2
def clean_data():
    global df
    print('='*50 + ' Question 2 ' + '='*50)
    duplicates = df.duplicated().sum()
    print(f"Số lượng dòng trùng lặp trước khi xóa: {duplicates}")
    df = df.drop_duplicates()
    missing_income = df['MonthlyIncome'].isnull().sum()
    missing_years = df['YearsAtCompany'].isnull().sum()
    print(f"Số lượng missing values - MonthlyIncome: {missing_income}, YearsAtCompany: {missing_years}")
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['YearsAtCompany'] = df['YearsAtCompany'].fillna(df['YearsAtCompany'].median())

# Question 3
def group_dept_role():
    df['Attrition_Num'] = np.where(df['Attrition'] == 'Yes', 1, 0)
    grouped = df.groupby(['Department', 'JobRole']).agg(
        Avg_MonthlyIncome=('MonthlyIncome', 'mean'),
        Attrition_Rate=('Attrition_Num', 'mean')
    ).reset_index()
    print('='*50 + ' Question 3 ' + '='*50)
    print(grouped.round(4))

# Question 4
def income_per_year_feature():
    df['Income_Per_Year'] = np.where(
        df['TotalWorkingYears'] == 0, 
        0, 
        df['MonthlyIncome'] / df['TotalWorkingYears']
    )
    df['Attrition_Num'] = np.where(df['Attrition'] == 'Yes', 1, 0)
    correlation = df['Income_Per_Year'].corr(df['Attrition_Num'])
    print('='*50 + ' Question 4 ' + '='*50)
    print("Đã tạo cột 'Income_Per_Year'.")
    print(f"Hệ số tương quan (Correlation) giữa Income_Per_Year và Attrition: {correlation:.4f}")

# Question 5
def visualize_age_travel():
    print('='*50 + ' Question 5 ' + '='*50)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='BusinessTravel', hue='Attrition', palette='Set2')
    plt.title('Attrition across Business Travel Frequencies')
    plt.xlabel('Business Travel')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# Question 6
def pivot_job_role():
    pivot = pd.pivot_table(
        df, 
        values='JobSatisfaction', 
        index='JobRole', 
        columns='Attrition', 
        aggfunc='mean'
    )
    print('='*50 + ' Question 6 ' + '='*50)
    print(pivot.round(2))

# Question 7
def experience_groups():
    df['Attrition_Num'] = np.where(df['Attrition'] == 'Yes', 1, 0)
    bins = [-1, 5, 10, 20, 100] 
    labels = ['0-5', '6-10', '11-20', '20+']
    df['Experience_Group'] = pd.cut(df['TotalWorkingYears'], bins=bins, labels=labels)
    attrition_by_exp = df.groupby('Experience_Group', observed=True)['Attrition_Num'].mean().reset_index()
    attrition_by_exp.columns = ['Experience_Group', 'Attrition_Rate']
    print('='*50 + ' Question 7 ' + '='*50)
    print(attrition_by_exp.round(4))

# Question 8
def segment_earners():
    median_income = df['MonthlyIncome'].median()
    df['Earner_Segment'] = np.where(df['MonthlyIncome'] >= median_income, 'High Earners', 'Low Earners')
    wlb_comparison = df.groupby('Earner_Segment')['WorkLifeBalance'].mean().reset_index()  
    print('='*50 + ' Question 8 ' + '='*50)
    print(f"Mức lương Median: {median_income}")
    print(wlb_comparison.round(2))

# Question 9
def advanced_visualizations():
    print('='*50 + ' Question 9 ' + '='*50)
    plt.figure(figsize=(15, 6))
    num_cols = df.select_dtypes(include=[np.number])
    plt.subplot(1, 2, 1)
    selected_num_cols = num_cols[['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance']]
    sns.heatmap(selected_num_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap')
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='Attrition', y='MonthlyIncome', palette='pastel')
    plt.title('MonthlyIncome by Attrition Status')
    plt.tight_layout()
    plt.show()

# Question 10
def feature_engineering():
    df_encoded = pd.get_dummies(df, columns=['Department', 'EducationField'], drop_first=True)
    df_encoded['OverTime_Encoded'] = df_encoded['OverTime'].map({'Yes': 1, 'No': 0})
    print('='*50 + ' Question 10 ' + '='*50)
    cols_to_show = ['OverTime', 'OverTime_Encoded'] + [col for col in df_encoded.columns if 'Department_' in col or 'EducationField_' in col]
    print("\nKết quả sau khi Encoding:")
    print(df_encoded[cols_to_show].head())

def main():
    load_and_inspect()
    clean_data()
    group_dept_role()
    income_per_year_feature()
    visualize_age_travel()
    pivot_job_role()
    experience_groups()
    segment_earners()
    advanced_visualizations()
    feature_engineering()

if __name__ == '__main__':
    main()
