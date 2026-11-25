import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

def knn_predict(x, k):
    dist = np.sqrt(((Xtr - x) ** 2).sum(axis=1))      # Euclidean distance
    nearest = np.argsort(dist)[:k]
    return Counter(ytr[nearest]).most_common(1)[0][0]  # majority vote

for k in [1, 3, 5, 7, 9]:
    pred = [knn_predict(x, k) for x in Xte]
    print(f"k = {k}: accuracy = {accuracy_score(yte, pred):.4f}")

pred = knn_predict(Xte[0], 5)
print("\nSample 0 -> predicted:", pred, " actual:", yte[0])
