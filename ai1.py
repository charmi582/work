from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x=[[5], [6], [7], [8], [9], [1], [2], [3], [4]]
y=[1, 1, 1, 1, 1, 0, 0, 0, 0]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test, y_pred)
print("準確率:" ,acc)

print("如果我上一餐吃了2 這餐吃得下嗎", model.predict([[2]]))

