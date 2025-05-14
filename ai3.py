from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x=[[5, 0], [12, 1], [3, 0], [15, 1], [7, 0], [10, 1], [4, 0], [8, 1]]
y=[0, 1, 0, 1, 0, 1, 0, 1]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)

model=LogisticRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_pred, y_test)
print("準確率:", acc)

now_data=[[9, 1]]
result=model.predict(now_data)[0]
if result==1:
    print("預測為:他不會買")
elif result==0:
    print("預測為:他會買")