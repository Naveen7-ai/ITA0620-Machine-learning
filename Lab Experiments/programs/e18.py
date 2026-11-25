from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

ppn = Perceptron(max_iter=1000, eta0=0.1, random_state=1).fit(Xtr, ytr)
pred = ppn.predict(Xte)
print("Weights shape:", ppn.coef_.shape, " Bias:", ppn.intercept_.round(3))
print("Training accuracy: %.4f" % ppn.score(Xtr, ytr))
print("Test accuracy    : %.4f" % accuracy_score(yte, pred))
print("Confusion matrix:\n", confusion_matrix(yte, pred))
print(classification_report(yte, pred, target_names=iris.target_names))
