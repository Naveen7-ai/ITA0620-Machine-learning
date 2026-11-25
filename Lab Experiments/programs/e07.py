import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_DIR = r"C:\Users\NAVEEN\OneDrive\Desktop\ML_Lab_Datasets_and_Programs\datasets"

df = pd.read_csv(DATA_DIR + r"\study_hours.csv")
print(df.head(), "\n")
X, y = df[["Hours"]], df["Pass"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

model = LogisticRegression().fit(Xtr, ytr)
pred = model.predict(Xte)
print("Coefficient: %.4f  Intercept: %.4f" % (model.coef_[0][0], model.intercept_[0]))
print("Accuracy: %.4f" % accuracy_score(yte, pred))
print("Confusion matrix:\n", confusion_matrix(yte, pred))

for h in [2, 5, 8]:
    p = model.predict_proba(pd.DataFrame({"Hours": [h]}))[0][1]
    print(f"Study hours = {h}: P(pass) = {p:.3f} -> {'Pass' if p > 0.5 else 'Fail'}")
