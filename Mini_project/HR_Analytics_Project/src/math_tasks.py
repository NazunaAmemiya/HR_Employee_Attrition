import numpy as np
import pandas as pd
from utils import get_data
from sklearn.linear_model import Lasso, Ridge 

df = get_data()

# Question 1
def matrix_rank():
    X = df[['Age', 'MonthlyIncome', 'TotalWorkingYears']].to_numpy()
    shape = X.shape
    rank = np.linalg.matrix_rank(X)
    print('='*50 + ' Question 1 ' + '='*50)
    print(f"Shape của Ma trận X: {shape}")
    print(f"Rank của Ma trận X: {rank}")

# Question 2
def linear_regression_attrition():
    y = np.where(df['Attrition'] == 'Yes', 1, 0)
    X = df[['Age', 'TotalWorkingYears']].to_numpy()
    X_b = np.c_[X, np.ones((X.shape[0], 1))]
    w_b = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    w = w_b[:2]
    b = w_b[2]
    print('='*50 + ' Question 2 ' + '='*50)
    print(f"Trọng số w (Age, TotalWorkingYears): {w.round(4)}")
    print(f"Bias b: {b:.4f}")

# Question 3
def probabilities():
    p_attrition = (df['Attrition'] == 'Yes').mean()
    overtime_yes_df = df[df['OverTime'] == 'Yes']
    p_attr_given_overtime = (overtime_yes_df['Attrition'] == 'Yes').mean()
    print('='*50 + ' Question 3 ' + '='*50)
    print(f"Xác suất nghỉ việc P(Attrition=1): {p_attrition:.4f}")
    print(f"Xác suất nghỉ việc khi có làm thêm P(Attrition=1 | OverTime='Yes'): {p_attr_given_overtime:.4f}")

# Question 4
def mse_gradient():
    y = np.where(df['Attrition'] == 'Yes', 1, 0)
    X = df[['Age', 'TotalWorkingYears']].to_numpy()
    n = len(y)
    w = np.zeros(X.shape[1])
    y_pred = X.dot(w)
    gradient = (-2 / n) * X.T.dot(y - y_pred)
    print('='*50 + ' Question 4 ' + '='*50)
    print("Công thức đạo hàm: ∇w L = (-2/n) * X^T * (y - Xw)")
    print(f"Giá trị Gradient khởi tạo tại w=0: {gradient.round(4)}")

# Question 5
def var_cov():
    income = df['MonthlyIncome'].to_numpy()
    years = df['YearsAtCompany'].to_numpy()
    variance = np.var(income, ddof=1)
    covariance = np.cov(income, years)[0, 1]
    print('='*50 + ' Question 5 ' + '='*50)
    print(f"Phương sai (Variance) của MonthlyIncome: {variance:,.2f}")
    print(f"Hiệp phương sai (Covariance) giữa Income và YearsAtCompany: {covariance:,.2f}")

# Question 6
def eigendecomposition():
    X = df[['Age', 'MonthlyIncome', 'YearsAtCompany']].to_numpy()
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    print('='*50 + ' Question 6 ' + '='*50)
    print("Trị riêng (Eigenvalues):")
    print(eigenvalues.round(2))
    print("\nVector riêng (Eigenvectors):")
    print(eigenvectors.round(4))

# Question 7
def svd_analysis():
    X = df[['Age', 'MonthlyIncome', 'TotalWorkingYears']].to_numpy()
    X_centered = X - np.mean(X, axis=0)
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
    print('='*50 + ' Question 7 ' + '='*50)
    print(f"Kích thước U (Mối quan hệ nhân viên - Đặc điểm ngầm): {U.shape}")
    print(f"Giá trị S (Mức độ quan trọng của Đặc điểm ngầm): {S.round(2)}")
    print(f"Kích thước VT (Mối quan hệ đặc trưng gốc - Đặc điểm ngầm): {VT.shape}")

#Question 8
def gradient_descent_optimization():
    X = df['Age'].to_numpy()
    y = df['MonthlyIncome'].to_numpy()
    X_scaled = (X - np.mean(X)) / np.std(X)
    y_scaled = (y - np.mean(y)) / np.std(y)
    n = len(y_scaled)
    w, b = 0.0, 0.0
    learning_rate = 0.1
    epochs = 100
    loss_history = []
    for i in range(epochs):
        y_pred = X_scaled * w + b
        error = y_scaled - y_pred
        dw = (-2 / n) * np.sum(X_scaled * error)
        db = (-2 / n) * np.sum(error)
        w -= learning_rate * dw
        b -= learning_rate * db
        mse = np.mean(error**2)
        if i % 20 == 0 or i == epochs - 1:
            loss_history.append(f"Epoch {i}: MSE Loss = {mse:.4f}")
    print('='*50 + ' Question 8 ' + '='*50)
    print(f"Trọng số tối ưu: w = {w:.4f}, b = {b:.4f}")
    print("\n".join(loss_history))

# Question 9
def bayes_theorem():
    df['Income_bin'] = pd.qcut(df['MonthlyIncome'], q=3, labels=['Low', 'Medium', 'High'])
    p_b = (df['Income_bin'] == 'Low').mean()
    p_a_and_b = ((df['Attrition'] == 'Yes') & (df['Income_bin'] == 'Low')).mean()
    p_a_given_b = p_a_and_b / p_b
    print('='*50 + ' Question 9 ' + '='*50)
    print(f"Xác suất Lương thấp P(Income='Low'): {p_b:.4f}")
    print(f"Xác suất Lương thấp & Nghỉ việc P(Attr='Yes' & Income='Low'): {p_a_and_b:.4f}")
    print(f"Xác suất Bayes P(Attr='Yes' | Income='Low'): {p_a_given_b:.4f}")

# Question 10
def regularization_comparison():
    X = df[['Age', 'MonthlyIncome']].to_numpy()
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = np.where(df['Attrition'] == 'Yes', 1, 0)
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_scaled, y)
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_scaled, y)
    result_df = pd.DataFrame({
        'Feature': ['Age', 'MonthlyIncome'],
        'Lasso (L1) Weights': lasso.coef_,
        'Ridge (L2) Weights': ridge.coef_
    })
    print('='*50 + ' Question 10 ' + '='*50)
    print(result_df.round(4))

def main():
    matrix_rank()
    linear_regression_attrition()
    probabilities()
    mse_gradient()
    var_cov()
    eigendecomposition()
    svd_analysis()
    gradient_descent_optimization()
    bayes_theorem()
    regularization_comparison()

if __name__ == '__main__':
    main()
