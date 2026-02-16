import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 0, 0, 1])

weights = np.zeros(2)
bias = 0
learning_rate = 0.1
epochs = 10

def step_function(x):
    return 1 if x > 0 else 0

for epoch in range(epochs):
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        y_pred = step_function(linear_output)
        error = y[i] - y_pred

        weights += learning_rate * error * X[i]
        bias += learning_rate * error

print('Финальные веса:', weights)
print('Финальное смещение:', bias)

predictions = [step_function(np.dot(x, weights) + bias) for x in X]
print('Предсказания:', predictions)
