import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Create simple dataset
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [10,20,30,40,50,60,70,80,90,100]
}

df = pd.DataFrame(data)

X = df[["Hours_Studied"]]
y = df["Marks"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score = r2_score(y_test, y_pred) * 100
print("Accuracy of Linear Regression model is %.2f%%" % score)
