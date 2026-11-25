import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_DIR = r"C:\Users\NAVEEN\OneDrive\Desktop\ML_Lab_Datasets_and_Programs\datasets"

df = pd.read_csv(DATA_DIR + r"\car_price.csv")
print(df.head(), "\n")
df = pd.get_dummies(df, columns=["Brand", "Fuel"], drop_first=True)
X, y = df.drop(columns="PriceLakh"), df["PriceLakh"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

for name, m in [("Linear Regression", LinearRegression()),
                ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=42))]:
    p = m.fit(Xtr, ytr).predict(Xte)
    print(f"{name:18s} R2 = {r2_score(yte, p):.4f}  MAE = {mean_absolute_error(yte, p):.3f}  RMSE = {np.sqrt(mean_squared_error(yte, p)):.3f}")

rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(Xtr, ytr)
new = pd.DataFrame([{c: 0.0 for c in X.columns}])
new.loc[0, ["Year", "KmDriven", "EngineL", "Brand_Honda"]] = [2018.0, 45000.0, 1.5, 1.0]
print("\nPredicted price of 2018 Honda, 45000 km, 1.5L petrol: %.2f lakh" % rf.predict(new)[0])
