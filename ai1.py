from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x=[[5], [6], [7], [8], [9], [1], [2], [3], [4]]
y=[1, 1, 1, 1, 1, 0, 0, 0, 0]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test, y_pred)
print("準確率:" ,acc)

print("睡 6.8 小時會精神嗎？預測為：", model.predict([[6.8]]))

