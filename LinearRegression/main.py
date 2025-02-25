import numpy as np


class MyLinearRegression:
    def __init__(
        self,
        n_iterations=1000,
        learning_rate=0.01,
        method='mini_batch_gradient',
        batch_size=None,
        regularization=None,
        reg_parameter=None,
        rho=None,
        random_state=42,
    ):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.method = method
        self.batch_size = batch_size
        self.regularization = regularization
        self.reg_parameter = reg_parameter
        self.rho = rho

    methods = {
        'ordinary_least_square': "ordinary_least_square",
        'batch_gradient': "batch_gradient_descent",
        'mini_batch_gradient': "mini_batch_gradient_descent",
        'stochastic_gradient': "stochastic_gradient",
    }

    # print(self.method)
    # if self.method not in methods:
    #     raise ValueError(
    #         f"The method is incorrect or not provided, Please choose from {list(methods.keys())}"
    #     )

    def fit(self, features, labels):
        # makeing sure that ones columns is stacked
        X = np.column_stack((features, np.ones(features.shape[0])))
        y = labels.copy()
        if self.method == "ordinary_least_square":
            self.theta = self.ordinary_least_square(X, y)
        elif self.method == "batch_gradient_descent":
            self.theta = self.compute_batch_gradient_descent(X, y)
        elif self.method == "mini_batch_gradient_descent":
            self.theta = self.compute_mini_batch_gradient_descent(X, y)
        elif self.method == "stochastic_gradient_descent":
            self.theta = self.compute_stochastic_gradient_descent(X, y)

    def predict(self, features):
        X_new = np.column_stack((features, np.ones(features.shape[0])))
        return X_new @ self.theta

    def ordinary_least_square(self, X, y):
        return np.linalg.inv(X.T @ X) @ X.T @ y

    def ordinary_least_square_pinv(self, X, y):
        return np.linalg.pinv(X) @ y  # check

    def compute_batch_gradient_descent(self, X, y):
        theta = np.random.randn(X.shape[1])
        for _ in range(self.n_iterations):
            theta -= self.learning_rate * self.loss_gradient(X, y, theta)
        return theta

    def compute_mini_batch_gradient_descent(self, X, y):
        theta = np.random.randn(X.shape[1])
        for _ in range(self.n_iterations):
            random_numbers = np.random.randint(0, len(y), self.batch_size)
            X_mini_batch = X[random_numbers]
            y_mini_batch = y[random_numbers]
            theta -= self.learning_rate * self.loss_gradient(
                X_mini_batch, y_mini_batch, theta
            )
        return theta

    def compute_stochastic_gradient_descent(self, X, y):
        theta = np.random.randn(X.shape[1])
        for _ in range(self.n_iterations):
            indices = np.random.permutation(len(y))
            for i in indices:
                X_i = X[i : i + 1]
                y_i = y[i : i + 1]

            theta -= self.learning_rate * self.loss_gradient(X_i, y_i, theta)
        return theta

    def loss_gradient(self, X, y, theta):
        error = X @ theta - y
        return (2 / len(y)) * X.T @ error

    # def l1_gradient(self, theta):
    #     return self.reg_parameter * np.sign(theta)

    # def l2_gradient(self, theta):
    #     return 2 * self.reg_parameter * theta

    # if self.regularization is None:

    #     def compute_gradients(self, X, y, theta):
    #         return self.loss_gradient(X, y, theta)

    # elif self.regularization == "lasso":

    #     def compute_gradients(self, X, y, theta):
    #         return self.loss_gradient(X, y, theta) + self.l1_gradient(theta)

    # elif self.regularization == "ridge":

    #     def compute_gradients(self, X, y, theta):
    #         return self.loss_gradient(X, y, theta) + self.l2_gradient(theta)

    # elif self.regularization == "elsastic_net":

    #     def compute_gradients(self, X, y, theta):
    #         loss_g = self.loss_gradient(X, y, theta)
    #         l1_g = self.l1_gradient(theta)
    #         l2_g = self.l2_gradient(theta)
    #         return loss_g + self.rho * (l1_g) * (1 - self.rho) * l2_g

    def mean_squared_error(self, y, y_pred):
        return np.sum((y_pred - y) ** 2) / len(y)


from sklearn.datasets import make_regression

X, y = make_regression(n_samples=200, n_features=20, noise=10, random_state=42)


# my_lin_reg = MyLinearRegression(method="ordinary_least_square")
# my_lin_reg = MyLinearRegression(
#     n_iterations=1000, method="batch_gradient_descent"
# )

# my_lin_reg = MyLinearRegression(
#     n_iterations=1000,
#     method="mini_batch_gradient_descent",
#     batch_size=30,
# )

my_lin_reg = MyLinearRegression(
    n_iterations=1000, method="stochastic_gradient_descent"
)
my_lin_reg.fit(X, y)
# print(my_lin_reg.theta)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
# print(lin_reg.intercept_)
# print(lin_reg.coef_)

X_new = X[1:5]

# print(X_new)
print(X_new)
print("My Linear Regression", my_lin_reg.predict(X_new))
print("sklearn lin reg:", lin_reg.predict(X_new))
print("True values:", y[1:5])
