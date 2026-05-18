# 📊 Dự án Phân tích Dữ liệu Nhân sự (HR Attrition Analysis) - Mini Project 2
Dự án này nhằm mục đích khám phá và phân tích các yếu tố dẫn đến quyết định nghỉ việc của nhân viên (Employee Attrition) dựa trên bộ dữ liệu mô phỏng của IBM. Dự án tập trung vào việc áp dụng các kỹ thuật tiền xử lý dữ liệu, đại số tuyến tính, xác suất thống kê và trực quan hóa dữ liệu bằng Python.

## 📖 Mục lục
- [Giới thiệu bộ dữ liệu](#-giới-thiệu-bộ-dữ-liệu)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Các mô-đun chính](#-các-mô-đun-chính)
- [Yêu cầu hệ thống và Cài đặt](#-yêu-cầu-hệ-thống-và-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)

## 📁 Giới thiệu bộ dữ liệu
- **Nguồn:** IBM HR Analytics Employee Attrition & Performance
- **Kích thước:** 1,470 dòng và 35 đặc trưng (features).
- **Biến mục tiêu (Target Variable):** `Attrition` (Yes/No) - Cho biết nhân viên đã rời khỏi tổ chức hay chưa.
- **Nội dung:** Chứa các thông tin về nhân khẩu học, mức lương (MonthlyIncome), thâm niên (TotalWorkingYears, YearsAtCompany), mức độ hài lòng với công việc và môi trường.

## 🗂 Cấu trúc dự án
Dự án được tổ chức theo chuẩn quy trình khoa học dữ liệu:
```text
HR-analytics-project/
├── data/
│   ├── raw/                # Chứa file dữ liệu gốc (hr.csv)
│   └── processed/          # Chứa dữ liệu đã qua làm sạch (nếu có)
├── notebooks/
│   └── hr_analysis.ipynb   # File Jupyter Notebook báo cáo tổng hợp
├── src/
│   ├── numpy_tasks.py      # Code giải quyết Module 1 (NumPy)
│   ├── math_tasks.py       # Code giải quyết Module 2 (Math for AI)
│   ├── pandas_tasks.py     # Code giải quyết Module 3 (Pandas & Visualization)
│   └── utils.py            # Các hàm hỗ trợ (nếu có)
├── outputs/
│   ├── figures/            # Chứa các biểu đồ xuất ra từ code
│   └── tables/             # Chứa kết quả tính toán dạng bảng
├── report/
│   └── final_report.pdf    # Báo cáo tóm tắt insight và kết luận
├── requirements.txt        # Danh sách thư viện cần thiết
└── README.md               # File tài liệu này
```
## 🛠 Các mô-đun chính
Dự án được chia thành 3 phần cốt lõi:
1. **Module 1: NumPy & Vectorization**
   - Trích xuất dữ liệu, kiểm tra shape, dtypes.
   - Các phép tính thống kê (Mean, Median, Std).
   - Chuẩn hóa dữ liệu (Min-Max Scaling, Z-score).
   - Tính toán khoảng cách (Euclidean), độ tương tự (Cosine Similarity) và PCA bằng code tay.
2. **Module 2: Mathematics for AI**
   - Biểu diễn ma trận dữ liệu và tìm Hạng (Rank).
   - Triển khai thuật toán Hồi quy tuyến tính (Linear Regression) bằng Normal Equation và Gradient Descent.
   - Phân tích Xác suất, Định lý Bayes, Covariance Matrix và Phân tích thành phần suy biến (SVD).
   - Áp dụng Regularization (L1 Lasso, L2 Ridge).
3. **Module 3: Pandas & Data Visualization**
   - Data cleaning (xóa lặp, điền khuyết).
   - Gom nhóm (Groupby), Feature Engineering (One-Hot Encoding, Label Encoding).
   - Phân tích trực quan với Matplotlib và Seaborn (Histogram, Bar chart, Heatmap, Boxplot).

## ⚙️ Yêu cầu hệ thống và Cài đặt
1. Đảm bảo máy tính của bạn đã cài đặt **Python 3.8+**.
2. Clone hoặc tải mã nguồn dự án về máy.
3. Mở Terminal / Command Prompt tại thư mục gốc của dự án và chạy lệnh sau để cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## 🚀 Hướng dẫn sử dụng
Bạn có thể chạy độc lập từng module bằng cách mở Terminal (hoặc Command Prompt), di chuyển vào thư mục dự án và gõ các lệnh tương ứng dưới đây:

**1. Kiểm tra Module 1 (NumPy & Vectorization):**
```bash
python HR_Analytics_Project/src/numpy_tasks.py
```
**2. Kiểm tra Module 2 (Toán học cho AI):**
```bash
python HR_Analytics_Project/src/math_tasks.py
```
**3. Kiểm tra Module 3 (Pandas & Trực quan hóa):**
```bash
python HR_Analytics_Project/src/pandas_tasks.py
```
