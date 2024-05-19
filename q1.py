import numpy as np
from cvxopt import matrix, solvers

# Given transformed data points and labels
X_transformed = np.array([[-3, 30], [-2, 0], [-1, -2], [0, 0], [1, -2], [2, 0], [3, 30]])
y = np.array([1, 1, -1, 1, -1, 1, 1])

# Calculate the kernel (in this case, the linear kernel of the transformed space)
N = len(y)
K = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        K[i,j] = np.dot(X_transformed[i], X_transformed[j])

# Set up the matrices for the quadratic programming problem
P = matrix(np.outer(y, y) * K)
q = matrix(-np.ones(N))
G = matrix(np.vstack((-np.eye(N), np.eye(N))))
h = matrix(np.hstack((np.zeros(N), np.ones(N) * np.inf)))
A = matrix(y, (1, N), 'd')
b = matrix(0.0)

# Solve the quadratic programming problem
sol = solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

# Extract support vectors and calculate w
w = sum(alphas[i] * y[i] * X_transformed[i] for i in range(N))

# Calculate w0 using a support vector
for i in range(N):
    if alphas[i] > 1e-4: # Arbitrary threshold to identify support vectors
        w0 = y[i] - np.dot(w, X_transformed[i])
        break

# Calculate the margin
margin = 2 / np.linalg.norm(w)

print("w:", w)
print("w0:", w0)
print("Margin:", margin)
