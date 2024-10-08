import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
# Tải dữ liệu
train_data = pd.read_csv('../De-bai/train_data.txt', header=None)
test_data = pd.read_csv('../De-bai/test_data.txt', header=None)

# Kiểm tra và loại bỏ các hàng không hợp lệ
train_data = train_data.apply(pd.to_numeric, errors='coerce').dropna()
test_data = test_data.apply(pd.to_numeric, errors='coerce').dropna()

# Tách biến mục tiêu và các đặc trưng
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data

# Phân tích tương quan (có thể bỏ qua nếu không cần thiết)
correlation_matrix = train_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Chia dữ liệu huấn luyện thành tập huấn luyện và tập kiểm tra
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Cân bằng dữ liệu với SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_split, y_train_split)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#Khởi tạo mô hình với các tham số
model = LogisticRegression(max_iter=5000, solver='saga', class_weight='balanced')  # Tăng số vòng lặp tối đa và cân bằng trọng số

# Huấn luyện mô hình
model.fit(X_train_balanced, y_train_balanced)

# Dự đoán và đánh giá
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Logistic Regression Accuracy: {accuracy:.2f}')

# Dự đoán trên dữ liệu kiểm tra
y_test_pred = model.predict(X_test)

# Lưu kết quả dự đoán vào tệp CSV
pd.DataFrame(y_test_pred, columns=['Predicted']).to_csv('logistic_regression_predictions.csv', index=False, header=True)

# Tối ưu hóa mô hình
param_grid = {
  'C': [0.1, 1, 10, 100],
  'solver': ['liblinear', 'saga', 'newton-cg'],
  'penalty': ['l1', 'l2']
}
grid_search = GridSearchCV(LogisticRegression(max_iter=5000), param_grid, cv=3)
grid_search.fit(X_train_balanced, y_train_balanced)

print(f'Best parameters for Logistic Regression: {grid_search.best_params_}')

# Đánh giá mô hình bằng k-fold cross-validation
cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5)
print(f'Cross-Validation Accuracy: {cv_scores.mean():.2f}')