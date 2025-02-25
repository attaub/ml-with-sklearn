import matplotlib.pyplot as plt
import numpy as np


class CustomLinearRegressor:
    def __init__(self, learning_rate=0.001, num_iterations=1000, method=''):
        self.num_iterations = num_iterations
        self.cost_history = []
        self.learning_path = []

    def train(self, X, y, method="batch_gradient_descent", *args, **kwargs):
        self.m = len(X)
        self.X_b = np.c_[np.ones((self.m, 1)), X]

        methods = {
            "batch_gradient_descent": self.batch_gradient_descent,
            "mini_batch_gradient_descent": self.mini_batch_gradient_descent,
            "stochastic_gradient_descent": self.stochastic_gradient_descent,
            "least_squares": self.least_squares_solution,
        }
        if method not in methods:
            raise ValueError(
                f"Invalid method '{method}'. Choose from {list(methods.keys())}."
            )

        self.theta = methods[method](X, y, *args, **kwargs)
        return self

    def predict(self, X_new):
        if self.theta is None:
            raise ValueError("Model is not trained. Call `train()` first.")
        X_new_b = np.c_[np.ones((len(X_new), 1)), X_new]
        return X_new_b.dot(self.theta)

    def batch_gradient_descent(self, X, y, n_iterations=1000, eta=0.1):
        theta = np.random.randn(self.X_b.shape[1], 1)
        for i in range(n_iterations):
            gradients = 2 / self.m * self.X_b.T.dot(self.X_b.dot(theta) - y)
            theta -= eta * gradients
            self.learning_path.append(theta)
            self.cost_history.append(self.compute_cost(X, y, theta))

        return theta

    def stochastic_gradient_descent(self, X, y, n_epochs=50, t0=5, t1=50):
        theta = np.random.randn(self.X_b.shape[1], 1)  # Dynamic shape

        def learning_schedule(t):
            return t0 / (t + t1)

        for epoch in range(n_epochs):
            for i in range(self.m):
                random_index = np.random.randint(self.m)
                xi = self.X_b[random_index : random_index + 1]
                yi = y[random_index : random_index + 1]
                gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
                eta_t = learning_schedule(epoch * self.m + i)
                theta -= eta_t * gradients
            self.learning_path.append(theta)
            self.cost_history.append(self.compute_cost(X, y, theta))

        return theta

    def mini_batch_gradient_descent(
        self, X, y, theta, alpha, iterations, batch_size
    ):
        m = len(y)
        cost_history = np.zeros(iterations)

        for i in range(iterations):
            indices = np.random.permutation(m)  # Shuffle data
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for j in range(0, m, batch_size):
                X_mini_batch = X_shuffled[
                    j : j + batch_size
                ]  # Select mini-batch
                y_mini_batch = y_shuffled[j : j + batch_size]

                gradient = (
                    X_mini_batch.T.dot(X_mini_batch.dot(theta) - y_mini_batch)
                    / batch_size
                )
                theta -= alpha * gradient  # Update parameters
            self.learning_path.append(theta)
            cost_history[i] = self.compute_cost(
                X, y, theta
            )  # updated cost history per iteration

        return theta

    def least_squares_solution(self, X, y):
        # Use `pinv` for stability
        return np.linalg.pinv(self.X_b.T.dot(self.X_b)).dot(self.X_b.T).dot(y)

    def compute_cost(self, X, y, theta):
        return (1 / (2 * len(y))) * np.sum((X.dot(theta) - y) ** 2)


class PolynomialRegressor:
    def __init__(
        self,
        degree=2,
        learning_rate=0.01,
        num_iterations=1000,
        method="batch_gradient_descent",
    ):
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.method = method
        self.theta = None
        self.cost_history = []

    def transform_features(self, X):
        """Transforms the features into polynomial features."""
        m = len(X)
        X_poly = np.ones(
            (m, 1)
        )  # Start with a column of ones (for the intercept term)
        for i in range(1, self.degree + 1):
            X_poly = np.c_[
                X_poly, X**i
            ]  # Add X^1, X^2, X^3,... up to X^degree
        return X_poly

    def fit(self, X, y):
        """Fits the model using the specified optimization method."""
        X_poly = self.transform_features(X)
        m = len(y)
        self.theta = np.random.randn(X_poly.shape[1], 1)

        # Dictionary of gradient descent methods
        methods = {
            "batch_gradient_descent": self.batch_gradient_descent,
            "stochastic_gradient_descent": self.stochastic_gradient_descent,
            "mini_batch_gradient_descent": self.mini_batch_gradient_descent,
        }

        # Ensure the selected method is valid
        if self.method not in methods:
            raise ValueError(f"Unknown method {self.method}")

        # Call the appropriate method and store cost history
        self.theta, self.cost_history = methods[self.method](X_poly, y, m)
        return self

    def batch_gradient_descent(self, X, y, m):
        """Batch gradient descent for polynomial regression."""
        cost_history = []
        for _ in range(self.num_iterations):
            gradients = (2 / m) * X.T.dot(X.dot(self.theta) - y)
            self.theta -= self.learning_rate * gradients
            cost = self.compute_cost(X, y)
            cost_history.append(cost)
        return self.theta, cost_history

    def stochastic_gradient_descent(self, X, y, m):
        """Stochastic gradient descent for polynomial regression."""
        cost_history = []
        for _ in range(self.num_iterations):
            for i in range(m):
                random_index = np.random.randint(m)
                xi = X[random_index : random_index + 1]
                yi = y[random_index : random_index + 1]
                gradients = 2 * xi.T.dot(xi.dot(self.theta) - yi)
                self.theta -= self.learning_rate * gradients
            cost = self.compute_cost(X, y)
            cost_history.append(cost)
        return self.theta, cost_history

    def mini_batch_gradient_descent(self, X, y, m, batch_size=32):
        """Mini-batch gradient descent for polynomial regression."""
        cost_history = []
        for _ in range(self.num_iterations):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, m, batch_size):
                X_mini_batch = X_shuffled[i : i + batch_size]
                y_mini_batch = y_shuffled[i : i + batch_size]
                gradients = (2 / batch_size) * X_mini_batch.T.dot(
                    X_mini_batch.dot(self.theta) - y_mini_batch
                )
                self.theta -= self.learning_rate * gradients
            cost = self.compute_cost(X, y)
            cost_history.append(cost)
        return self.theta, cost_history

    def compute_cost(self, X, y):
        """Computes the cost (mean squared error)."""
        m = len(y)
        return (1 / (2 * m)) * np.sum((X.dot(self.theta) - y) ** 2)

    def predict(self, X):
        """Makes predictions on new data."""
        X_poly = self.transform_features(X)
        return X_poly.dot(self.theta)

    def plot(self, X, y, X_new):
        """Plots the data points and the polynomial regression curve."""
        y_pred = self.predict(X_new)
        plt.scatter(X, y, color='blue', label='Original data')
        plt.plot(
            X_new,
            y_pred,
            color='red',
            label=f'{self.method} Polynomial degree {self.degree}',
        )
        plt.legend()
        plt.show()


import numpy as np
import matplotlib.pyplot as plt


class LassoRegressor:
    def __init__(
        self,
        degree=2,
        alpha=0.1,
        learning_rate=0.01,
        num_iterations=1000,
        method="coordinate_descent",
    ):
        self.degree = degree
        self.alpha = alpha  # Regularization strength (lambda)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.method = method
        self.theta = None
        self.cost_history = []

    def transform_features(self, X):
        """Transforms input X into polynomial features."""
        m = len(X)
        X_poly = np.ones((m, 1))  # Intercept term
        for i in range(1, self.degree + 1):
            X_poly = np.c_[X_poly, X**i]  # Add polynomial features
        return X_poly

    def fit(self, X, y):
        """Fits Lasso Regression using the selected optimization method."""
        X_poly = self.transform_features(X)
        m, n = X_poly.shape
        self.theta = np.random.randn(n, 1)

        methods = {
            "coordinate_descent": self.coordinate_descent,
            "gradient_descent": self.gradient_descent,
        }

        if self.method not in methods:
            raise ValueError(f"Unknown method {self.method}")

        self.theta, self.cost_history = methods[self.method](X_poly, y, m, n)
        return self

    def coordinate_descent(self, X, y, m, n):
        """Lasso regression using coordinate descent."""
        cost_history = []
        for _ in range(self.num_iterations):
            for j in range(n):
                X_j = X[:, j : j + 1]  # Extract j-th feature
                residual = (
                    y - X.dot(self.theta) + X_j.dot(self.theta[j])
                )  # Exclude j-th feature effect
                rho_j = X_j.T.dot(residual) / m  # Compute correlation

                if rho_j > self.alpha:
                    self.theta[j] = rho_j - self.alpha
                elif rho_j < -self.alpha:
                    self.theta[j] = rho_j + self.alpha
                else:
                    self.theta[j] = 0  # Set to zero if within threshold

            cost = self.compute_cost(X, y)
            cost_history.append(cost)

        return self.theta, cost_history

    def gradient_descent(self, X, y, m, n):
        """Lasso regression using gradient descent."""
        cost_history = []
        for _ in range(self.num_iterations):
            gradients = (2 / m) * X.T.dot(X.dot(self.theta) - y)
            lasso_term = self.alpha * np.sign(
                self.theta
            )  # L1 Regularization term
            self.theta -= self.learning_rate * (gradients + lasso_term)

            cost = self.compute_cost(X, y)
            cost_history.append(cost)

        return self.theta, cost_history

    def compute_cost(self, X, y):
        """Computes Lasso cost function (MSE + L1 regularization)."""
        m = len(y)
        mse = (1 / (2 * m)) * np.sum((X.dot(self.theta) - y) ** 2)
        l1_penalty = self.alpha * np.sum(np.abs(self.theta))
        return mse + l1_penalty

    def predict(self, X):
        """Predicts outputs for new inputs."""
        X_poly = self.transform_features(X)
        return X_poly.dot(self.theta)

    def plot(self, X, y, X_new):
        """Plots data and the fitted Lasso regression curve."""
        y_pred = self.predict(X_new)
        plt.scatter(X, y, color='blue', label='Original Data')
        plt.plot(X_new, y_pred, color='red', label=f'Lasso ({self.method})')
        plt.legend()
        plt.show()


import numpy as np
import matplotlib.pyplot as plt


class RidgeRegressor:
    def __init__(
        self,
        degree=2,
        alpha=0.1,
        learning_rate=0.01,
        num_iterations=1000,
        method="gradient_descent",
    ):
        self.degree = degree
        self.alpha = alpha  # Regularization strength (lambda)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.method = method
        self.theta = None
        self.cost_history = []

    def transform_features(self, X):
        """Transforms input X into polynomial features."""
        m = len(X)
        X_poly = np.ones((m, 1))  # Intercept term
        for i in range(1, self.degree + 1):
            X_poly = np.c_[X_poly, X**i]  # Add polynomial features
        return X_poly

    def fit(self, X, y):
        """Fits Ridge Regression using the selected optimization method."""
        X_poly = self.transform_features(X)
        m, n = X_poly.shape
        self.theta = np.random.randn(n, 1)

        methods = {
            "gradient_descent": self.gradient_descent,
            "normal_equation": self.normal_equation,
        }

        if self.method not in methods:
            raise ValueError(f"Unknown method {self.method}")

        self.theta, self.cost_history = methods[self.method](X_poly, y, m, n)
        return self

    def gradient_descent(self, X, y, m, n):
        """Ridge regression using gradient descent."""
        cost_history = []
        for _ in range(self.num_iterations):
            gradients = (2 / m) * X.T.dot(X.dot(self.theta) - y) + (
                2 * self.alpha * self.theta
            )
            self.theta -= self.learning_rate * gradients

            cost = self.compute_cost(X, y)
            cost_history.append(cost)

        return self.theta, cost_history

    def normal_equation(self, X, y, m, n):
        """Ridge regression using normal equation."""
        I = np.eye(n)  # Identity matrix
        I[0, 0] = 0  # Do not regularize the intercept term
        self.theta = np.linalg.inv(X.T.dot(X) + self.alpha * I).dot(X.T).dot(y)
        return self.theta, []

    def compute_cost(self, X, y):
        """Computes Ridge cost function (MSE + L2 regularization)."""
        m = len(y)
        mse = (1 / (2 * m)) * np.sum((X.dot(self.theta) - y) ** 2)
        l2_penalty = self.alpha * np.sum(self.theta**2)
        return mse + l2_penalty

    def predict(self, X):
        """Predicts outputs for new inputs."""
        X_poly = self.transform_features(X)
        return X_poly.dot(self.theta)

    def plot(self, X, y, X_new):
        """Plots data and the fitted Ridge regression curve."""
        y_pred = self.predict(X_new)
        plt.scatter(X, y, color='blue', label='Original Data')
        plt.plot(X_new, y_pred, color='red', label=f'Ridge ({self.method})')
        plt.legend()
        plt.show()


import numpy as np
import matplotlib.pyplot as plt


class ElasticNetRegressor:
    def __init__(
        self,
        degree=2,
        alpha=0.1,
        beta=0.5,
        learning_rate=0.01,
        num_iterations=1000,
        method="gradient_descent",
    ):
        """
        Elastic Net Regressor.
        degree: Polynomial degree for feature transformation.
        alpha: Regularization strength.
        beta: Mixing parameter (0 = Ridge, 1 = Lasso).
        learning_rate: Learning rate for gradient descent.
        num_iterations: Number of iterations for gradient descent.
        method: Optimization method ("gradient_descent" or "normal_equation").
        """
        self.degree = degree
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.method = method
        self.theta = None
        self.cost_history = []

    def transform_features(self, X):
        """Transforms input X into polynomial features."""
        m = len(X)
        X_poly = np.ones((m, 1))  # Bias term
        for i in range(1, self.degree + 1):
            X_poly = np.c_[X_poly, X**i]  # Add polynomial features
        return X_poly

    def fit(self, X, y):
        """Fits Elastic Net Regression using the selected optimization method."""
        X_poly = self.transform_features(X)
        m, n = X_poly.shape
        self.theta = np.random.randn(n, 1)

        methods = {
            "gradient_descent": self.gradient_descent,
            "normal_equation": self.normal_equation,
        }

        if self.method not in methods:
            raise ValueError(f"Unknown method {self.method}")

        self.theta, self.cost_history = methods[self.method](X_poly, y, m, n)
        return self

    def gradient_descent(self, X, y, m, n):
        """Elastic Net regression using gradient descent."""
        cost_history = []
        for _ in range(self.num_iterations):
            predictions = X.dot(self.theta)
            error = predictions - y

            # Compute gradients (L2 Ridge + L1 Lasso)
            l1_grad = self.beta * np.sign(self.theta)  # L1 derivative
            l2_grad = (1 - self.beta) * self.theta  # L2 derivative
            regularization = self.alpha * (l1_grad + 2 * l2_grad)

            gradients = (1 / m) * X.T.dot(error) + regularization
            self.theta -= self.learning_rate * gradients

            cost = self.compute_cost(X, y)
            cost_history.append(cost)

        return self.theta, cost_history

    def normal_equation(self, X, y, m, n):
        """Elastic Net regression using normal equation."""
        I = np.eye(n)
        I[0, 0] = 0  # Do not regularize the intercept term
        ridge_term = (1 - self.beta) * I  # Ridge (L2)
        lasso_term = self.beta * np.sign(self.theta)  # Lasso (L1)
        self.theta = (
            np.linalg.inv(X.T.dot(X) + self.alpha * ridge_term).dot(X.T).dot(y)
            - self.alpha * lasso_term
        )
        return self.theta, []

    def compute_cost(self, X, y):
        """Computes Elastic Net cost function (MSE + L1 + L2 regularization)."""
        m = len(y)
        mse = (1 / (2 * m)) * np.sum((X.dot(self.theta) - y) ** 2)
        l1_penalty = self.beta * np.sum(np.abs(self.theta))
        l2_penalty = (1 - self.beta) * np.sum(self.theta**2)
        return mse + self.alpha * (l1_penalty + l2_penalty)

    def predict(self, X):
        """Predicts outputs for new inputs."""
        X_poly = self.transform_features(X)
        return X_poly.dot(self.theta)

    def plot(self, X, y, X_new):
        """Plots data and the fitted Elastic Net regression curve."""
        y_pred = self.predict(X_new)
        plt.scatter(X, y, color='blue', label='Original Data')
        plt.plot(
            X_new, y_pred, color='red', label=f'Elastic Net (β={self.beta})'
        )
        plt.legend()
        plt.show()


from sklearn import datasets

X, y = datasets.make_regression(
    n_samples=1000, n_features=2, noise=10, random_state=42
)

cus_reg = CustomLinearRegressor(
    learning_rate=0.001, num_iterations=1000, method='least_squares'
)

cus_reg.train(X, y)

# def poincare_pseudo_inverse(A, tol=1e-6):
#     """
#     Compute the Poincaré pseudo-inverse of matrix A.

#     Parameters:
#         A (numpy.ndarray): The input matrix.
#         tol (float): Tolerance for singular values considered as zero.

#     Returns:
#         numpy.ndarray: The Poincaré pseudo-inverse of A.
#     """
#     U, S, Vt = np.linalg.svd(A, full_matrices=False)

#     # Compute reciprocal of nonzero singular values
#     S_inv = np.array([1/s if s > tol else 0 for s in S])

#     # Construct the pseudo-inverse
#     A_pseudo = Vt.T @ np.diag(S_inv) @ U.T
#     return A_pseudo

# # Example usage
# A = np.array([[1, 2], [2, 4]])  # Singular matrix
# A_pseudo = poincare_pseudo_inverse(A)

# print("Original Matrix A:")
# print(A)
# print("\nPoincaré Pseudo-Inverse of A:")
# print(A_pseudo)
