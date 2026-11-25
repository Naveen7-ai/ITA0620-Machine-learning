import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

DATA_DIR = r"C:\Users\NAVEEN\OneDrive\Desktop\ML_Lab_Datasets_and_Programs\datasets"

df = pd.read_csv(DATA_DIR + r"\position_salary.csv")
print(df, "\n")
X, y = df[["Level"]].values, df["Salary"].values

lin = LinearRegression().fit(X, y)
p_lin = lin.predict(X)

poly = PolynomialFeatures(degree=2)
Xp = poly.fit_transform(X)
pr = LinearRegression().fit(Xp, y)
p_poly = pr.predict(Xp)

print("Linear Regression      : R2 = %.4f  RMSE = %.2f" % (r2_score(y, p_lin), np.sqrt(mean_squared_error(y, p_lin))))
print("Polynomial (degree 2)  : R2 = %.4f  RMSE = %.2f" % (r2_score(y, p_poly), np.sqrt(mean_squared_error(y, p_poly))))
print("\nPrediction for level 6.5:")
print("  Linear     : %.2f" % lin.predict([[6.5]])[0])
print("  Polynomial : %.2f" % pr.predict(poly.transform([[6.5]]))[0])
