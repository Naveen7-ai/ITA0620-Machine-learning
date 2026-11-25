import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

DATA_DIR = r"C:\Users\NAVEEN\OneDrive\Desktop\ML_Lab_Datasets_and_Programs\datasets"

df = pd.read_csv(DATA_DIR + r"\salary.csv")
print(df.head(), "\n")
X, y = df[["YearsExperience"]], df["Salary"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression().fit(Xtr, ytr)
pred = model.predict(Xte)
print("Slope (coefficient): %.2f" % model.coef_[0])
print("Intercept          : %.2f" % model.intercept_)
print("MAE : %.2f" % mean_absolute_error(yte, pred))
print("MSE : %.2f" % mean_squared_error(yte, pred))
print("R2  : %.4f" % r2_score(yte, pred))
print("Predicted salary for 10 years experience: %.2f" % model.predict(pd.DataFrame({"YearsExperience": [10]}))[0])
