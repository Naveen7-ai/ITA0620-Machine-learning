import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(1)
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
Y = np.eye(3)[y]                                   # one-hot encoding
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=y)

sigmoid = lambda z: 1 / (1 + np.exp(-z))
W1 = np.random.randn(4, 8) * 0.5; b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 3) * 0.5; b2 = np.zeros((1, 3))
lr = 0.5

for epoch in range(1, 1001):
    # forward pass
    h = sigmoid(Xtr @ W1 + b1)
    out = sigmoid(h @ W2 + b2)
    loss = np.mean((Ytr - out) ** 2)
    # backward pass
    d_out = (out - Ytr) * out * (1 - out)
    d_h = (d_out @ W2.T) * h * (1 - h)
    W2 -= lr * h.T @ d_out / len(Xtr); b2 -= lr * d_out.mean(0, keepdims=True)
    W1 -= lr * Xtr.T @ d_h / len(Xtr); b1 -= lr * d_h.mean(0, keepdims=True)
    if epoch % 200 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}  MSE loss = {loss:.4f}")

def predict(X):
    return np.argmax(sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2), axis=1)

print("\nTrain accuracy: %.2f%%" % (100 * np.mean(predict(Xtr) == Ytr.argmax(1))))
print("Test accuracy : %.2f%%" % (100 * np.mean(predict(Xte) == Yte.argmax(1))))
