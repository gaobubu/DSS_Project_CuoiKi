import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv("/Users/dokhanhduy/Documents/HỌC TẬP/data.csv", delimiter=";")
data.columns = data.columns.str.strip()

# Chuyển đổi cột Order_Date thành datetime
data["Order_Date"] = pd.to_datetime(data["Order_Date"], dayfirst=True)

# Tính tổng doanh thu hàng ngày trong tháng 12/2019
data["Sales"] = data["Quantity_Ordered"] * data["Price_each"].str.replace(",", "").astype(float)
daily_sales = data.groupby("Order_Date")["Sales"].sum().reset_index()

# Chuẩn bị dữ liệu cho các mô hình
X = np.array((daily_sales["Order_Date"] - daily_sales["Order_Date"].min()).dt.days).reshape(-1, 1)
y = daily_sales["Sales"].values

# Hàm chạy mô hình Linear Regression
def run_linear_regression(X, y):
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_pred = linear_model.predict(np.array([[31]]))  # Dự đoán cho ngày 1/1/2020
    mae = mean_absolute_error(y, linear_model.predict(X))
    mse = mean_squared_error(y, linear_model.predict(X))
    return y_pred, mae, mse

# Hàm chạy mô hình Decision Tree Regressor
def run_decision_tree(X, y):
    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X, y)
    y_pred = tree_model.predict(np.array([[31]]))  # Dự đoán cho ngày 1/1/2020
    mae = mean_absolute_error(y, tree_model.predict(X))
    mse = mean_squared_error(y, tree_model.predict(X))
    return y_pred, mae, mse

# Chạy cả hai mô hình và lấy kết quả
y_pred_linear, mae_linear, mse_linear = run_linear_regression(X, y)
y_pred_tree, mae_tree, mse_tree = run_decision_tree(X, y)

# Tính phần trăm chính xác
mean_actual = y.mean()
accuracy_linear = (1 - mae_linear / mean_actual) * 100
accuracy_tree = (1 - mae_tree / mean_actual) * 100

# In ra kết quả độ chính xác
print("Kết quả Linear Regression:")
print(f"Dự đoán doanh thu cho ngày 1/1/2020: {y_pred_linear[0]:.2f}")
print(f"MAE: {mae_linear:.2f}, MSE: {mse_linear:.2f}, Độ chính xác: {accuracy_linear:.2f}%")

print("\nKết quả Decision Tree Regressor:")
print(f"Dự đoán doanh thu cho ngày 1/1/2020: {y_pred_tree[0]:.2f}")
print(f"MAE: {mae_tree:.2f}, MSE: {mse_tree:.2f}, Độ chính xác: {accuracy_tree:.2f}%")

# So sánh để tìm mô hình tốt hơn
if mae_linear < mae_tree and mse_linear < mse_tree:
    print("\nMô hình Linear Regression có độ chính xác tốt hơn dựa trên MAE và MSE.")
elif mae_tree < mae_linear and mse_tree < mse_linear:
    print("\nMô hình Decision Tree Regressor có độ chính xác tốt hơn dựa trên MAE và MSE.")
else:
    print("\nKhông có mô hình nào hoàn toàn tốt hơn. Hãy xem xét cả hai mô hình để lựa chọn.")

# Vẽ biểu đồ so sánh kết quả dự đoán của hai mô hình
plt.figure(figsize=(14, 7))
plt.plot(daily_sales["Order_Date"], y, marker='o', color="black", label="Doanh thu thực tế tháng 12/2019", linewidth=2)
plt.scatter(pd.Timestamp("2020-01-01"), y_pred_linear[0], color="blue", s=100, label="Dự đoán Linear Regression tháng 1/2020", edgecolor='black')
plt.scatter(pd.Timestamp("2020-01-01"), y_pred_tree[0], color="green", s=100, label="Dự đoán Decision Tree tháng 1/2020", edgecolor='black')

# Thêm thông tin cho biểu đồ
plt.xlabel("Ngày")
plt.ylabel("Doanh thu")
plt.title("Dự đoán doanh thu cho ngày 1/1/2020 giữa Linear Regression và Decision Tree Regressor")
plt.xticks(rotation=45)
plt.axhline(y=y.mean(), color='red', linestyle='--', label='Doanh thu trung bình tháng 12/2019')
plt.legend()
plt.grid(True)
plt.show()
