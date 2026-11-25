import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_DIR = r"C:\Users\NAVEEN\OneDrive\Desktop\ML_Lab_Datasets_and_Programs\datasets"

df = pd.read_csv(DATA_DIR + r"\bank_loan.csv")
print(df.head(), "\n")
print("Loan status counts:\n", df["LoanApproved"].value_counts(), "\n")
df = pd.get_dummies(df, columns=["Gender", "Married", "Education"], drop_first=True)
X, y = df.drop(columns="LoanApproved"), df["LoanApproved"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

nb = GaussianNB().fit(Xtr, ytr)
pred = nb.predict(Xte)
print("Accuracy: %.4f" % accuracy_score(yte, pred))
print("Confusion matrix:\n", confusion_matrix(yte, pred))
print(classification_report(yte, pred))
