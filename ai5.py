import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# 資料
X = np.array([[8.0, 4.0],[7.5, 5.0], [6.0, 8.0], [5.5, 9.0],
              [7.0, 6.0], [6.0, 7.5], [7.2, 5.5], [6.2, 8.5]])
y = np.array([1, 1, 0, 0, 1, 0, 1, 0])

# 建模
model = LogisticRegression()
model.fit(X, y)

# 畫圖
plt.figure(figsize=(8, 6))

# 畫資料點
for label in np.unique(y):
    plt.scatter(X[y == label][:, 0], X[y == label][:, 1],
                label=f"完成運動={label}", s=100)

# 畫分類邊界
x_min, x_max = X[:, 0].min()-0.5, X[:, 0].max()+0.5
y_min, y_max = X[:, 1].min()-0.5, X[:, 1].max()+0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlBu)
plt.xlabel("睡眠時數")
plt.ylabel("工作時數")
plt.title("AI 分類模型視覺化：是否完成運動")
plt.legend()
plt.grid(True)
plt.show()
