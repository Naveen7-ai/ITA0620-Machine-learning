import numpy as np
from scipy.stats import norm

np.random.seed(0)
# Data drawn from a mixture of two Gaussians
data = np.concatenate([np.random.normal(2, 1.0, 300), np.random.normal(8, 1.5, 200)])

# Initial guesses
mu = np.array([1.0, 6.0]); sigma = np.array([1.0, 1.0]); pi = np.array([0.5, 0.5])

for it in range(1, 51):
    # E-step: responsibilities
    resp = np.array([pi[k] * norm.pdf(data, mu[k], sigma[k]) for k in range(2)])
    resp /= resp.sum(axis=0)
    # M-step: update parameters
    Nk = resp.sum(axis=1)
    mu = (resp * data).sum(axis=1) / Nk
    sigma = np.sqrt((resp * (data - mu[:, None]) ** 2).sum(axis=1) / Nk)
    pi = Nk / len(data)
    ll = np.log(sum(pi[k] * norm.pdf(data, mu[k], sigma[k]) for k in range(2))).sum()
    if it in (1, 5, 10, 25, 50):
        print(f"Iter {it:2d}: mu = {mu.round(3)}, sigma = {sigma.round(3)}, pi = {pi.round(3)}, log-likelihood = {ll:.3f}")

print("\nFinal estimated parameters")
for k in range(2):
    print(f"Component {k+1}: mean = {mu[k]:.3f}, std = {sigma[k]:.3f}, weight = {pi[k]:.3f}")
print("True parameters: (2, 1.0, 0.6) and (8, 1.5, 0.4)")
