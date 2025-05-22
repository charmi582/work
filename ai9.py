from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler

x=np.array([[6, 8, 1], [5, 10, 2], [7, 7, 0], [4.5, 11, 3], [6.5, 9.5, 1], [7.5, 8, 0], [5, 12, 3], [8, 6.5, 0], [6, 10, 2], [5.5, 11.5, 2], [6.8, 7.5, 1], [4.8, 11.8, 3]])
y=np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1])

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

log_model=LogisticRegression()
log_model.fit(x_scaler, y)

knn_model=KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_scaler, y)

log_pred=log_model.predict(x_scaler)
knn_pred=knn_model.predict(x_scaler)

log_acc=accuracy_score(y, log_pred)
knn_acc=accuracy_score(y, knn_pred)
print("log準確率:", log_acc)
print("knn準確率:", knn_acc)


a=eval(input("請輸入睡眠時數:"))
b=eval(input("請輸入工作時數:"))
c=eval(input("請輸入喝了幾杯咖啡:"))

new_data=np.array([[a, b, c]])
now_data=scaler.transform(new_data)

result1=log_model.predict(now_data)[0]
print("log predict 您會加班" if result1==1 else "log predict 不會加班")
result2=knn_model.predict(now_data)[0]
print("knn predict 您會加班" if result2==1 else "knn predict不會加班")