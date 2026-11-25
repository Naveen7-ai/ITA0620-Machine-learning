from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

model = GaussianNB().fit(Xtr, ytr)
pred = model.predict(Xte)

print("Confusion matrix:\n", confusion_matrix(yte, pred))
print("\nAccuracy: %.4f" % accuracy_score(yte, pred))
print("\nClassification report:\n", classification_report(yte, pred, target_names=load_wine().target_names))
