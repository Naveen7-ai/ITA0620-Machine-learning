import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_DIR = r"C:\Users\NAVEEN\OneDrive\Desktop\ML_Lab_Datasets_and_Programs\datasets"

df = pd.read_csv(DATA_DIR + r"\house_price.csv")
print(df.describe().round(2), "\n")
X, y = df.drop(columns="PriceLakh"), df["PriceLakh"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

for name, m in [("Linear Regression", LinearRegression()),
                ("Gradient Boosting", GradientBoostingRegressor(random_state=42))]:
    p = m.fit(Xtr, ytr).predict(Xte)
    print(f"{name:18s} R2 = {r2_score(yte, p):.4f}  MAE = {mean_absolute_error(yte, p):.3f}  RMSE = {np.sqrt(mean_squared_error(yte, p)):.3f}")

gb = GradientBoostingRegressor(random_state=42).fit(Xtr, ytr)
house = pd.DataFrame([[1800, 3, 2, 5, 8.0]], columns=X.columns)
print("\nPredicted price for 1800 sq ft, 3 bed, 2 bath, 5 yrs old, 8 km: %.2f lakh" % gb.predict(house)[0])
