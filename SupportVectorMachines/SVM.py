import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

class SVM_SGD:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param  # Regularization parameter
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        y_ = np.where(y <= 0, -1, 1)  # Convert labels to -1 and 1
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

######################
# Generate toy dataset
X, y = make_classification(n_samples=200, n_features=2, n_classes=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1

######################
# Train SVM
svm = SVM_SGD(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
svm.fit(X, y)

######################
# Plot decision boundary
def plot_decision_boundary(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.show()

plot_decision_boundary(X, y, svm)

#################################################################
import numpy as np
import cvxopt
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

class SVM_QP:
    def __init__(self, C=1.0):
        self.C = C  # Regularization parameter
        self.w = None
        self.b = None
        self.support_vectors = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.astype(float)
        y[y == 0] = -1  # Convert labels to -1 and 1
        
        # Compute Gram matrix (Kernel function: Linear)
        K = np.dot(X, X.T)
        
        # Set up the Quadratic Programming problem for cvxopt solver
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n_samples))
        
        G_std = np.diag(-np.ones(n_samples))
        h_std = np.zeros(n_samples)
        
        G_slack = np.diag(np.ones(n_samples))
        h_slack = np.ones(n_samples) * self.C
        
        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.hstack((h_std, h_slack)))
        
        A = cvxopt.matrix(y, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)
        
        # Solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])
        
        # Support vectors have non-zero lagrange multipliers
        sv = alphas > 1e-5
        self.support_vectors = X[sv]
        self.w = np.sum(alphas[sv] * y[sv, np.newaxis] * X[sv], axis=0)
        self.b = np.mean(y[sv] - np.dot(X[sv], self.w))
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# Generate toy dataset
X, y = make_classification(n_samples=200, n_features=2, n_classes=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
y = np.where(y == 0, -1, 1)

# Train SVM using QP
svm = SVM_QP(C=1.0)
svm.fit(X, y)

# Plot decision boundary
def plot_decision_boundary(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.legend()
    plt.show()

plot_decision_boundary(X, y, svm)

#################################################################
class SVMClassifierOld:
    def __init__(self, kernel='linear', C=1.0, gamma=0.1):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = None
        self.b = None
        self.support_vectors = None
        self.support_labels = None
        self.weights = None  # Only used for linear kernel


    def fit(self, X, y):
        self.alpha, self.b = self.smo_svm(X, y, self.C)
        # Identify support vectors
        support_vector_mask = self.alpha > 1e-5
        self.support_vectors = X[support_vector_mask]
        self.support_labels = y[support_vector_mask]
        self.alpha = self.alpha[support_vector_mask]

        # Compute weight vector if using a linear kernel
        if self.kernel == 'linear':
            self.weights = np.sum(
                self.alpha[:, None]
                * self.support_labels[:, None]
                * self.support_vectors,
                axis=0,
            )

    def predict(self, X):
        if self.kernel == 'linear':
            decision_function = np.dot(X, self.weights) + self.b
        else:
            kernel_values = self.rbf_kernel(X, self.support_vectors)
            decision_function = (
                np.dot(kernel_values, self.alpha * self.support_labels)
                + self.b
            )

        return np.sign(decision_function)

    def linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def rbf_kernel(self, X1, X2):
        sq_dists = (
            np.sum(X1**2, axis=1).reshape(-1, 1)
            + np.sum(X2**2, axis=1)
            - 2 * np.dot(X1, X2.T)
        )
        return np.exp(-self.gamma * sq_dists)

    def smo_svm(self, X, y, C=1.0, tol=1e-3, max_passes=5):
        n_samples, n_features = X.shape
        alpha = np.zeros(n_samples)
        b = 0
        passes = 0

        # Select kernel function dynamically
        if self.kernel == 'linear':
            kernel = self.linear_kernel(X, X)
        elif self.kernel == 'rbf':
            kernel = self.rbf_kernel(X, X)
        else:
            raise ValueError("Unsupported kernel")

        while passes < max_passes:
            num_changed_alphas = 0
            for i in range(n_samples):
                Ei = np.dot((alpha * y), kernel[:, i]) + b - y[i]
                if (y[i] * Ei < -tol and alpha[i] < C) or (
                    y[i] * Ei > tol and alpha[i] > 0
                ):
                    j = np.random.choice(
                        [x for x in range(n_samples) if x != i]
                    )
                    Ej = np.dot((alpha * y), kernel[:, j]) + b - y[j]

                    alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(C, C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - C)
                        H = min(C, alpha[i] + alpha[j])

                    if L == H:
                        continue

                    eta = 2 * kernel[i, j] - kernel[i, i] - kernel[j, j]
                    if eta >= 0:
                        continue

                    alpha[j] -= (y[j] * (Ei - Ej)) / eta
                    alpha[j] = np.clip(alpha[j], L, H)

                    if abs(alpha[j] - alpha_j_old) < tol:
                        continue

                    alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])

                    b1 = (
                        b
                        - Ei
                        - y[i] * (alpha[i] - alpha_i_old) * kernel[i, i]
                        - y[j] * (alpha[j] - alpha_j_old) * kernel[i, j]
                    )
                    b2 = (
                        b
                        - Ej
                        - y[i] * (alpha[i] - alpha_i_old) * kernel[i, j]
                        - y[j] * (alpha[j] - alpha_j_old) * kernel[j, j]
                    )

                    if 0 < alpha[i] < C:
                        b = b1
                    elif 0 < alpha[j] < C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

                    num_changed_alphas += 1

            passes = passes + 1 if num_changed_alphas == 0 else 0

        return alpha, b

