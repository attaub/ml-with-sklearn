import numpy as np
from sklearn.linear_model import LogisticRegression


class MyLogisticRegression:
    def __init__(self, n_iterations=1000, learning_rate=0.005):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.coefficients = None
        self.intercept = None
        self.N = None
        self.D = None

    def fit(self, X, y, method="bgd"):
        self.N, self.D = X.shape
        self.coefficients = np.zeros(self.D)
        self.intercept = 0
        if method == "bgd":
            self.batch_gradient_descent(X, y)
        if method == "sgd":
            self.stochastic_gradient_descent(X, y)

    def batch_gradient_descent(self, X, y):
        for _ in range(self.n_iterations):
            logits = np.dot(X, self.coefficients) + self.intercept
            probs = self.sigmoid(logits)
            error = probs - y

            gradients_weights = (1 / self.N) * np.dot(X.T, error)
            gradients_bias = (1 / self.N) * np.sum(error)

            self.coefficients -= self.learning_rate * gradients_weights
            self.intercept -= self.learning_rate * gradients_bias

    def stochastic_gradient_descent(self, X, y):
        for _ in range(self.n_iterations):
            indices = np.random.permutation(len(y))

            for i in indices:
                X_i = X[i : i + 1]
                y_i = y[i : i + 1]

                logits = np.dot(X_i, self.coefficients) + self.intercept
                probs = self.sigmoid(logits)

                gradients_weights = (probs - y_i) * X_i
                gradients_bias = probs - y_i

                self.coefficients -= self.learning_rate * gradients_weights
                self.intercept -= self.learning_rate * gradients_bias

    def predict(self, X):
        logits = np.dot(X, self.coefficients) + self.intercept
        probs = self.sigmoid(logits)
        return [0 if prob <= 0.5 else 1 for prob in probs]

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))


################################################################################

from sklearn.datasets import make_classification


# Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=2,
    n_redundant=2,
    n_classes=2,
    random_state=42,
)

# Check the shape of the generated dataset
print(X.shape)  # (1000, 20)
print(y.shape)  # (1000,)


my_clf = MyLogisticRegression()
my_clf.fit(X, y)
print()
print(my_clf.coefficients)
print(my_clf.intercept)

my_clf_2 = MyLogisticRegression()
# my_clf_2.fit(X, y, method="sgd")
# print()
# print(my_clf_2.coefficients)
# print(my_clf_2.intercept)
# print("=====")

from sklearn.linear_model import LogisticRegression

skl_clf = LogisticRegression()
skl_clf.fit(X, y)
print()
print(skl_clf.coef_)
print(skl_clf.intercept_)


# pridectinon

i = 50
X_new = X[i : i + 5]
y_new = y[i : i + 5]

print()
print(y_new)
print()
print(my_clf.predict(X_new))
print(skl_clf.predict(X_new))
