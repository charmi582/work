from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler

x=np.array([[7, 8], [6, 9], [8, 7], [5, 11], [6.5, 10], [7.5, 8], [4.5, 12], [8, 6], [6, 9.5], [5.5, 11]])
y=np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0])

scaler = StandardScaler()
x_scaler=scaler.fit_transform(x)

log_model=LogisticRegression()
knn_model=KNeighborsClassifier(n_neighbors=3)

log_model.fit(x_scaler, y)
knn_model.fit(x_scaler, y)

log_pred=log_model.predict(x_scaler)
knn_pred=knn_model.predict(x_scaler)

log_acc=accuracy_score(y, log_pred)
knn_acc=accuracy_score(y, knn_pred)
print("log_model準確率", log_acc)
print("knn準確率:", knn_acc)

n=eval(input("請輸入您的睡眠時間"))
b=eval(input("請輸入您的工作時間"))

now_data=np.array([[n, b]])
new_data=scaler.transform(now_data)
result1=log_model.predict(new_data)[0]
print("您會運動"if result1==1 else "您不會運動")
result2=knn_model.predict(new_data)[0]
print("您會運動"if result1==1 else "您不會運動")
