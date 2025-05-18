from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x=[[8.0, 4.0],[7.5, 5.0], [6.0, 8.0], [5.5, 9.0], [7.0, 6.0], [6.0, 7.5], [7.2, 5.5], [6.2, 8.5]]
y=[1, 1, 0, 0, 1, 0, 1, 0]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

model=LogisticRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test, y_pred)
print("準確率:", acc)

now_data=[[6.5, 5.5]]
result=model.predict(now_data)[0]
print("會完成運動目標"if result==1 else "不會完成目標")