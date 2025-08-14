import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class LogisticNeuron:
    def __init__(self, input_dim, learning_rate=0.1, epochs=1000):
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []

    def tanh(self, z):
        ### START CODE HERE ###
        ### TODO: implement the tanh activation
        z = np.clip(z, -500, 500)         # estabilidade numérica
        a = np.tanh(z)
        ### END CODE HERE ###
        return a

    def predict_proba(self, X):
        ### START CODE HERE ###
        ### TODO: compute activation output using tanh
        z = X @ self.weights + self.bias
        a = self.tanh(z)                  # saída em (-1, 1)
        ### END CODE HERE ###
        return a

    def predict(self, X):
        # Classe {0,1} com limiar em 0 sobre a saída tanh
        prediction = (self.predict_proba(X) >= 0).astype(int)
        return prediction

    def train(self, X, y):
        ### START CODE HERE ###
        ### TODO: convert y from {0, 1} to {-1, +1}
        y = y.reshape(-1).astype(float)
        y_tanh = 2.0 * y - 1.0

        m = X.shape[0]
        for _ in range(self.epochs):
            # Forward pass
            y_pred = self.predict_proba(X)          # (-1,1)

            # Compute error
            error = y_pred - y_tanh                 # (m,)

            # Gradients (para MSE e tanh: delta = (a - y)*tanh'(z) = (a - y)*(1-a^2))
            delta = error * (1.0 - y_pred**2)       # (m,)
            grad_w = (X.T @ delta) / m              # (n,)
            grad_b = np.sum(delta) / m              # escalar

            # Update parameters
            self.weights -= self.learning_rate * grad_w
            self.bias    -= self.learning_rate * grad_b

            # Compute MSE loss: (1/(2m)) * sum (a - y)^2
            loss = (1.0 / (2.0 * m)) * np.sum(error**2)
            self.loss_history.append(loss)
        ### END CODE HERE ###

def generate_dataset():
    X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=2.0)
    return X, y

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Tanh Output (-1..1)')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title('Tanh Activation Decision Boundary')
    plt.show()

def plot_loss(model):
    plt.plot(model.loss_history, 'k.')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('MSE Loss over Training Iterations')
    plt.show()

def main():
    # Generate dataset
    X, y = generate_dataset()

    # Train the model
    neuron = LogisticNeuron(input_dim=2, learning_rate=0.1, epochs=100)
    neuron.train(X, y)

    # Plot decision boundary
    plot_decision_boundary(neuron, X, y)

    # Plot loss over training iterations
    plot_loss(neuron)

if __name__ == "__main__":
    main()
