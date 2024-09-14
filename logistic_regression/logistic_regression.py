import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Bước 1: Tải dữ liệu
train_data = pd.read_csv('../De-bai/train_data.txt', header=None)
test_data = pd.read_csv('../De-bai/test_data.txt', header=None)

# Kiểm tra và loại bỏ các hàng không hợp lệ
train_data = train_data.apply(pd.to_numeric, errors='coerce').dropna()
test_data = test_data.apply(pd.to_numeric, errors='coerce').dropna()

# Tách biến mục tiêu và các đặc trưng
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data

# Bước 2: Chia dữ liệu huấn luyện thành tập huấn luyện và tập kiểm tra
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_split = scaler.fit_transform(X_train_split)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Bước 3: Khởi tạo mô hình
model = LogisticRegression(max_iter=2000)

# Huấn luyện mô hình
model.fit(X_train_split, y_train_split)

# Dự đoán và đánh giá
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Logistic Regression Accuracy: {accuracy:.2f}')

# Dự đoán trên dữ liệu kiểm tra
y_test_pred = model.predict(X_test)

# Lưu kết quả dự đoán vào tệp
pd.DataFrame(y_test_pred).to_csv('logistic_regression_predictions.csv', index=False, header=False)

# Tối ưu hóa mô hình
param_grid = {'C': [0.1, 1, 10]}
grid_search = GridSearchCV(LogisticRegression(max_iter=2000), param_grid, cv=3)
grid_search.fit(X_train_split, y_train_split)

print(f'Best parameters for Logistic Regression: {grid_search.best_params_}')