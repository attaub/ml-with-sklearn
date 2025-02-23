import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

x = np.random.uniform(0, 10, 100)
noise = np.random.normal(0, 2, 100)
y = 2 * x + noise

dataset = np.column_stack((x, y))

plt.scatter(x, y, label="Data Points")
plt.xlabel("Feature (x)")
plt.ylabel("Target (y)")
plt.title("Synthetic Dataset")
plt.legend()
plt.show()

lin_reg = CustomLinearRegressor(
    learning_rate=0.001,
    num_iterations=1000,
    method="batch_gradient_descent",
)
