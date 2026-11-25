import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = r"C:\Users\NAVEEN\OneDrive\Desktop\ML_Lab_Datasets_and_Programs\datasets"

df = pd.read_csv(DATA_DIR + r"\credit_score.csv")
print("Shape:", df.shape)
print(df.head(), "\n")
print("Class distribution:\n", df["CreditScore"].value_counts(), "\n")

X, y = df.drop(columns="CreditScore"), df["CreditScore"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
pred = model.predict(Xte)
print("Accuracy: %.4f\n" % accuracy_score(yte, pred))
print(classification_report(yte, pred))
print("Confusion matrix:\n", confusion_matrix(yte, pred, labels=["Good", "Standard", "Poor"]))
print("\nFeature importance:")
for f, i in sorted(zip(X.columns, model.feature_importances_), key=lambda t: -t[1]):
    print(f"  {f:24s} {i:.4f}")
