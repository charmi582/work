from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score

x=np.array([[7, 8], [6, 9], [8, 7], [5, 11], [6.5, 10], [7.5, 8], [4.5, 12], [8, 6], [6, 9.5], [5.5, 11]])
y=np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0])

model=LogisticRegression()
model.fit(x, y)

y_pred=model.predict(x)
acc=accuracy_score(y, y_pred)
print("準確度:", acc)
print("請輸入您的工作時間:")
a=eval(input())
print("請輸入您的睡眠時間")
b=eval(input())
c=a+b
while c<24:
    now_data=[[b, a]]
    result=model.predict(now_data)[0]
    print("會運動"if result==1 else "不會運動")
if c>=24:
    print("輸入異常")