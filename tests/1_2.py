import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_samples = 200
X = np.random.uniform(-5, 5, n_samples)
noise = np.random.normal(0, 0.5, n_samples)

y = -1.5 * X + 2 + noise

lr0 = 0.1
decay_rate = 0.01
epochs = 100
w = 5.0
b = 5.0

loss_history = []

for epoch in range(epochs):
    lr = lr0 / (1 + decay_rate * epoch)

    indices = np.random.permutation(n_samples)
    for i in indices:
        xi = X[i]
        yi = y[i]

        y_pred = w * xi + b
        error = y_pred - yi

        dw = 2 * error * xi
        db = 2 * error

        w -= lr * dw
        b -= lr * db
    
    y_pred_all = w * X + b
    loss = np.mean((y_pred_all - y) ** 2)
    loss_history.append(loss)

plt.figure(figsize=(10, 8))
plt.plot(loss_history, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Сходимость SGD с убывающим learning rate')
plt.grid()
plt.show()

print(f"Финальные параметры: w = {w:.4f}, b = {b:.4f}")
print(f"Финальный loss: {loss:.6f}")
