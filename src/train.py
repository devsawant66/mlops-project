from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import joblib

X, y = load_diabetes(return_X_y=True)

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Training Complete")
