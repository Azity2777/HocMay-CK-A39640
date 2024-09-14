import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

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

# Phân tích tương quan
correlation_matrix = train_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#Chia dữ liệu huấn luyện thành tập huấn luyện và tập kiểm tra
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_split = scaler.fit_transform(X_train_split)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#Khởi tạo mô hình với các tham số
model = DecisionTreeClassifier(max_depth=10, min_samples_split=5) 

# Huấn luyện mô hình
model.fit(X_train_split, y_train_split)

# Dự đoán và đánh giá
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Decision Tree Accuracy: {accuracy:.2f}')

# Dự đoán trên dữ liệu kiểm tra
y_test_pred = model.predict(X_test)

# Lưu kết quả dự đoán vào tệp
pd.DataFrame(y_test_pred).to_csv('decision_tree_predictions.csv', index=False, header=False)

# Tối ưu hóa mô hình
param_grid = {
  'max_depth': [None, 5, 10, 15, 20],  
  'min_samples_split': [2, 5, 10, 15], 
  'min_samples_leaf': [1, 2, 5]  
}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3)
grid_search.fit(X_train_split, y_train_split)

print(f'Best parameters for Decision Tree: {grid_search.best_params_}')