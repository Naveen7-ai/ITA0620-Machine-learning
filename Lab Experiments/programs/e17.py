import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_DIR = r"C:\Users\NAVEEN\OneDrive\Desktop\ML_Lab_Datasets_and_Programs\datasets"

df = pd.read_csv(DATA_DIR + r"\mobile_price.csv")
print(df.head(), "\n")
print("Price range distribution:\n", df["PriceRange"].value_counts().sort_index(), "\n")
X, y = df.drop(columns="PriceRange"), df["PriceRange"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
pred = rf.predict(Xte)
print("Accuracy: %.4f" % accuracy_score(yte, pred))
print("Confusion matrix:\n", confusion_matrix(yte, pred))
print(classification_report(yte, pred))
phone = pd.DataFrame([[3500, 1500, 1600, 32, 8, 16]], columns=X.columns)
print("Predicted price range for a 3500MB RAM phone:", rf.predict(phone)[0], "(0=low ... 3=very high)")
