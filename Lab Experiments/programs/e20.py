import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_DIR = r"C:\Users\NAVEEN\OneDrive\Desktop\ML_Lab_Datasets_and_Programs\datasets"

df = pd.read_csv(DATA_DIR + r"\advertising.csv")
print(df.head(), "\n")
print("Correlation with Sales:\n", df.corr()["Sales"].round(3), "\n")
X, y = df[["TV", "Radio", "Newspaper"]], df["Sales"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(Xtr, ytr)
pred = model.predict(Xte)
print("Coefficients:", {c: round(float(v), 4) for c, v in zip(X.columns, model.coef_)}, " Intercept: %.4f" % model.intercept_)
print("R2 = %.4f  MAE = %.4f  RMSE = %.4f" % (r2_score(yte, pred), mean_absolute_error(yte, pred), np.sqrt(mean_squared_error(yte, pred))))

future = pd.DataFrame({"TV": [150, 250], "Radio": [20, 40], "Newspaper": [30, 50]})
future["PredictedSales"] = model.predict(future).round(2)
print("\nFuture sales forecast for planned ad budgets:\n", future)
