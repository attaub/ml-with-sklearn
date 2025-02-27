import numpy as np


class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0, gamma=0.1):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = None
        self.b = None
        self.support_vectors = None
        self.support_labels = None
        self.weights = None  # Only used for linear kernel

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
