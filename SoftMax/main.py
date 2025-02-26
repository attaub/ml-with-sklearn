import numpy as np

class SoftmaxClassifier:
    def __init__(self, input_dim, num_classes, learning_rate=0.1):
        self.W = np.random.randn(input_dim, num_classes) * 0.01  
        self.b = np.zeros((1, num_classes))  
        self.lr = learning_rate

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def cross_entropy_loss(self, Y_true, Y_pred):
        N = Y_true.shape[0]
        loss = -np.sum(Y_true * np.log(Y_pred + 1e-9)) / N  
        return loss

    def compute_gradients(self, X, Y_true, Y_pred):
        N = X.shape[0]
        dZ = (Y_pred - Y_true) / N
        dW = np.dot(X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        return dW, db

    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            Z = np.dot(X, self.W) + self.b  
            Y_pred = self.softmax(Z)  
            loss = self.cross_entropy_loss(Y, Y_pred)  

            dW, db = self.compute_gradients(X, Y, Y_pred)

            self.W -= self.lr * dW
            self.b -= self.lr * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        Z = np.dot(X, self.W) + self.b
        Y_pred = self.softmax(Z)
        return np.argmax(Y_pred, axis=1)  

# usage
if __name__ == "__main__":
    np.random.seed(42)
    
    X_train = np.random.rand(5, 3)
    Y_labels = np.array([0, 1, 2, 3, 1])  
    Y_train = np.eye(4)[Y_labels]  # One-hot encoding

    model = SoftmaxClassifier(input_dim=3, num_classes=4)
    model.train(X_train, Y_train, epochs=1000)

    predictions = model.predict(X_train)
    print("Predicted Classes:", predictions)
