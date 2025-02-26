import numpy as np

class MyLogisticRegression:
    def __init__(self, n_iterations=1000, learning_rate=0.005):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.coefficients = None  # Includes the intercept term
        self.N = None  # Number of samples
        self.D = None  # Number of features

    def fit(self, X, y, method="bgd"):
        self.N, self.D = X.shape
        X = np.c_[X, np.ones(self.N)]  
        self.coefficients = np.zeros(self.D + 1)  

        if method == "bgd":
            self.batch_gradient_descent(X, y)
        elif method == "sgd":
            self.stochastic_gradient_descent(X, y)
        else:
            raise ValueError("Invalid method. Choose 'bgd' or 'sgd'.")

    def batch_gradient_descent(self, X, y):
        for _ in range(self.n_iterations):
            logits = np.dot(X, self.coefficients)  
            probs = self.sigmoid(logits)
            error = probs - y

            gradients = (1 / self.N) * np.dot(X.T, error)  
            self.coefficients -= self.learning_rate * gradients

    def stochastic_gradient_descent(self, X, y):
        for _ in range(self.n_iterations):
            indices = np.random.permutation(len(y))
            for i in indices:
                X_i = X[i]
                y_i = y[i]

                logits = np.dot(X_i, self.coefficients)  
                probs = self.sigmoid(logits)

                gradients = (probs - y_i) * X_i  
                self.coefficients -= self.learning_rate * gradients

    def predict(self, X):
        X = np.c_[X, np.ones(X.shape[0])]  
        logits = np.dot(X, self.coefficients)  
        probs = self.sigmoid(logits)
        return [0 if prob <= 0.5 else 1 for prob in probs]

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))
