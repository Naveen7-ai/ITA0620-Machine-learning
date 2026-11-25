from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC()),
}
print(f"{'Model':20s}{'Acc':>8s}{'Prec':>8s}{'Recall':>8s}{'F1':>8s}{'CV-5':>8s}")
for name, m in models.items():
    p = m.fit(Xtr, ytr).predict(Xte)
    cv = cross_val_score(m, X, y, cv=5).mean()
    print(f"{name:20s}{accuracy_score(yte,p):8.4f}{precision_score(yte,p):8.4f}{recall_score(yte,p):8.4f}{f1_score(yte,p):8.4f}{cv:8.4f}")
