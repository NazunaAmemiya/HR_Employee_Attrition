import numpy as np
import pandas as pd
from utils import get_data

df = get_data()

# Question 1
def shape_and_dtypes():
    X_axis = df[['Age','MonthlyIncome','TotalWorkingYears']].values
    print('='*50 + ' Question 1 ' + '='*50)
    print(f"Shape: {X_axis.shape} \nType: {X_axis.dtype}")

#Question 2
def med():
    X_axis = df[['Age','MonthlyIncome']].to_numpy()
    means = np.mean(X_axis, axis = 0)
    median = np.median(X_axis, axis = 0)
    deviation = np.std(X_axis, axis = 0)
    result = pd.DataFrame([means, median, deviation], 
                        columns=['Age', 'MonthlyIncome'],
                        index=['Mean', 'Median', 'Std']
                        )
    print('='*50 + ' Question 2 ' + '='*50)
    print(result.round(2))

#Question 3
def normalization():
    X_axis = df[['MonthlyIncome','TotalWorkingYears']].to_numpy()
    x_min = np.min(X_axis,axis=0)
    x_max = np.max(X_axis,axis=0)
    normal = (X_axis - x_min) / (x_max - x_min)
    norm = pd.DataFrame(normal,columns=['MonthlyIncome','TotalWorkingYears'],index=range(1, len(df) + 1))
    print('='*50 + ' Question 3 ' + '='*50)
    print(norm.round(2))

# Question 4
def filter_():
    X_axis = df[['Age','MonthlyIncome']].to_numpy()
    fil = X_axis[(X_axis[:,0] > 35) & (X_axis[:,1] < np.mean(X_axis[:,1]))]
    filtered = pd.DataFrame(fil,columns=['Age','MonthlyIncome'],index=range(1,len(fil)+1))
    print('='*50 + ' Question 4 ' + '='*50)
    print(filtered)

# Question 5
def matrix():
    X_axis = df[['Age','MonthlyIncome','YearsAtCompany']]
    mtx = np.corrcoef(X_axis,rowvar=False)
    mtx_df = pd.DataFrame(mtx,['Age','MonthlyIncome','YearsAtCompany'],['Age','MonthlyIncome','YearsAtCompany'])
    print('='*50 + ' Question 5 ' + '='*50)
    print(mtx_df.round(2))

# Question 6
def euclidean_distance():
    X = df[['MonthlyIncome', 'TotalWorkingYears']].to_numpy()
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    dist_df = pd.DataFrame(dist_matrix[:5, :5], 
                           columns=[f'NV_{i}' for i in range(5)], 
                           index=[f'NV_{i}' for i in range(5)])
    print('='*50 + ' Question 6 ' + '='*50)
    print("Ma trận khoảng cách Euclidean (5 nhân viên đầu tiên):")
    print(dist_df.round(2))

# Question 7
def compare_normalization():
    income = df['MonthlyIncome'].to_numpy()
    min_val = np.min(income)
    max_val = np.max(income)
    min_max_scaled = (income - min_val) / (max_val - min_val)
    mean_val = np.mean(income)
    std_val = np.std(income)
    z_score_scaled = (income - mean_val) / std_val
    compare_df = pd.DataFrame({
        'Income_Gốc': income[:5],
        'Min_Max_Scaled': min_max_scaled[:5],
        'Z_Score_Scaled': z_score_scaled[:5]
    })
    
    print('='*50 + ' Question 7 ' + '='*50)
    print(compare_df.round(4))

# Question 8
def cosine_sim():
    features = ['Age', 'MonthlyIncome', 'TotalWorkingYears']
    X = df[features].to_numpy()
    nv_A = X[0]
    nv_B = X[1]
    dot_product = np.dot(nv_A, nv_B)
    norm_A = np.linalg.norm(nv_A)
    norm_B = np.linalg.norm(nv_B)
    cos_similarity = dot_product / (norm_A * norm_B)
    print('='*50 + ' Question 8 ' + '='*50)
    print(f"Vector Nhân viên A (Dòng 0): {nv_A}")
    print(f"Vector Nhân viên B (Dòng 1): {nv_B}")
    print(f"-> Độ tương tự Cosine: {cos_similarity:.4f} (Càng gần 1 thì càng giống nhau)")

# Question 9
def manual_pca():
    features = ['Age', 'MonthlyIncome', 'TotalWorkingYears']
    X = df[features].to_numpy()
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    top_2_components = sorted_eigenvectors[:, :2]
    X_pca = np.dot(X_centered, top_2_components)
    pca_df = pd.DataFrame(X_pca[:5, :], columns=['PC1', 'PC2'])
    print('='*50 + ' Question 9 ' + '='*50)
    print("Dữ liệu sau khi giảm xuống còn 2 chiều (Top 2 Principal Components):")
    print("Hiển thị 5 dòng đầu:")
    print(pca_df.round(2))

# Question 10
def batch_processing():
    features = ['Age', 'MonthlyIncome', 'TotalWorkingYears']
    X = df[features].to_numpy()
    batch_size = 50
    batch_means = []
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size]
        mean_vector = np.mean(batch, axis=0)
        batch_means.append(mean_vector)
    batch_df = pd.DataFrame(batch_means, columns=features)
    print('='*50 + ' Question 10 ' + '='*50)
    print(f"Tổng số dữ liệu: {len(X)} nhân viên.")
    print(f"Tổng số lô tạo ra (kích thước {batch_size}): {len(batch_df)} lô.")
    print("Hiển thị giá trị trung bình của 5 lô đầu tiên:")
    print(batch_df.head(5).round(2))

def main():
    shape_and_dtypes()
    med()
    normalization()
    filter_()
    matrix()
    euclidean_distance()
    compare_normalization()
    cosine_sim()
    manual_pca()
    batch_processing()

if __name__ == '__main__':
    main()
