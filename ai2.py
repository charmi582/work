from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x=[[8], [7.5], [7], [6.5], [6], [6.5], [5], [5.5], [5], [4.5], [4]]
y=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

x_train, x_test, y_train, y_test=train_test_split(test_size=0.3)

model=LogisticRegression
model.fix(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test, y_pred)
print(f"準確率:{acc}")

print(f"我上一晚睡了這麼多，這樣有睡飽嗎", acc)