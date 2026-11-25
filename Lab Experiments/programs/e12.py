from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)

for k in [1, 3, 5, 7, 9, 11]:
    acc = accuracy_score(yte, KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr).predict(Xte))
    print(f"k = {k:2d}  accuracy = {acc:.4f}")

knn = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
pred = knn.predict(Xte)
print("\nChosen k = 5, accuracy = %.4f" % accuracy_score(yte, pred))
print("Confusion matrix:\n", confusion_matrix(yte, pred))
print(classification_report(yte, pred, target_names=iris.target_names))
sample = [[5.1, 3.5, 1.4, 0.2]]
print("Sample", sample[0], "-> predicted:", iris.target_names[knn.predict(sample)[0]])
