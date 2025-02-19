import matplotlib.pyplot as plt
import numpy as np

class MyLinearRegressor:
    def __init__(self, X, y):
    # def __init__(self, X, y, learning_rate=0.001, num_iterations=1000, method=''):
        self.X = X
        self.y = y
        self.m = len(X)
        self.X_b = np.c_[np.ones((self.m, 1)), X]  # Fixed shape issue
        self.theta = None  # Initialize theta

    def train(self, method="batch_gradient_descent", *args, **kwargs):
        methods = {
            "batch_gradient_descent": self.batch_gradient_descent,
            "stochastic_gradient_descent": self.stochastic_gradient_descent,
            "least_squares": self.least_squares_solution,
            "least_squares": self.least_squares_solution,
        }
        if method not in methods:
            raise ValueError(
                f"Invalid method '{method}'. Choose from {list(methods.keys())}."
            )
        self.theta = methods[method](*args, **kwargs)
        return self  # Enables method chaining

    def predict(self, X_new):
        if self.theta is None:
            raise ValueError("Model is not trained. Call `train()` first.")
        X_new_b = np.c_[np.ones((len(X_new), 1)), X_new]
        return X_new_b.dot(self.theta)

    def batch_gradient_descent(self, n_iterations=1000, eta=0.1):
        theta = np.random.randn(
            self.X_b.shape[1], 1
        )  # Dynamic size for multiple features
        for _ in range(n_iterations):
            gradients = (
                2 / self.m * self.X_b.T.dot(self.X_b.dot(theta) - self.y)
            )
            theta -= eta * gradients
        return theta

    def stochastic_gradient_descent(self, n_epochs=50, t0=5, t1=50):
        theta = np.random.randn(self.X_b.shape[1], 1)  # Dynamic shape

        def learning_schedule(t):
            return t0 / (t + t1)

        for epoch in range(n_epochs):
            for i in range(self.m):
                random_index = np.random.randint(self.m)
                xi = self.X_b[random_index : random_index + 1] # gives an array
                yi = self.y[random_index : random_index + 1]
                gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
                eta_t = learning_schedule(epoch * self.m + i)
                theta -= eta_t * gradients

        return theta

    def least_squares_solution(self):
        return (
            np.linalg.pinv(self.X_b.T.dot(self.X_b))
            .dot(self.X_b.T)
            .dot(self.y)
        )  # Use `pinv` for stability


def plot_data_first(X, y):
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.show()


def plot_predictions_1(X, y, X_new, y_predict):
    plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([0, 2, 0, 15])
    plt.show()


def plot_gradient_descent(X_b, X, y, X_new, theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)


def plot_bgd(X_b, X, y, X_new, theta, theta_path_bgd):
    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plot_gradient_descent(X_b, X, y, X_new, theta, eta=0.02)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(132)
    plot_gradient_descent(
        X_b, X, y, X_new, theta, eta=0.1, theta_path=theta_path_bgd
    )
    plt.subplot(133)
    plot_gradient_descent(X_b, X, y, X_new, theta, eta=0.5)

    plt.show()
