import numpy as np


class LinearRegressionLoss:
    """
    Mean Squared Error (MSE) loss function for linear regression.
    L(y, y_pred) = (1 / (2n)) * sum((y_pred - y)^2)
    """

    def loss(self, y_true, y_pred):
        n = y_true.shape[0]
        return np.sum((y_pred - y_true) ** 2) / (2.0 * n)

    def grad(self, X, y_true, y_pred):
        """
        Compute gradients of the loss function with respect to weights and bias.
        """
        n = X.shape[0]
        # Partial derivative with respect to y_pred
        dl_dy_pred = (y_pred - y_true) / n
        # Partial derivative with respect to weights (using chain rule)
        dl_dw = X.T @ dl_dy_pred
        # Partial derivative with respect to bias
        dl_db = np.sum(dl_dy_pred)

        return dl_dw, dl_db


class GradientDescent:
    """
    Batch Gradient Descent optimizer.
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6, verbose=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y, loss_fn):
        """
        Train the linear regression model.
        """
        n, d = X.shape

        # 1. Initialize parameters
        self.w = np.zeros(d)
        self.b = 0.0

        # 2. Training loop
        for i in range(self.max_iter):
            # Forward pass: compute predictions
            y_pred = X @ self.w + self.b

            # Compute current loss
            current_loss = loss_fn.loss(y, y_pred)
            self.loss_history.append(current_loss)

            # Backward pass: compute gradients
            dl_dw, dl_db = loss_fn.grad(X, y, y_pred)

            # Update parameters
            self.w -= self.learning_rate * dl_dw
            self.b -= self.learning_rate * dl_db

            # Print training logs (optional)
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i:4d} | Loss: {current_loss:.6f}")

            # 3. Early stopping condition
            if np.linalg.norm(dl_dw) < self.tol and np.abs(dl_db) < self.tol:
                if self.verbose:
                    print(f"Converged early at iteration {i}.")
                break

    def predict(self, X):
        """
        Predict target values for given input data.
        """
        return X @ self.w + self.b


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data (200 samples, 3 features)
    n, d = 200, 3
    X = np.random.randn(n, d)
    true_w = np.array([2.0, -3.0, 0.5])
    true_b = 1.5
    # y = Xw + b + noise
    y = X @ true_w + true_b + 0.2 * np.random.randn(n)

    # Initialize loss function and optimizer
    loss_fn = LinearRegressionLoss()
    optimizer = GradientDescent(learning_rate=0.1, max_iter=2000, tol=1e-6, verbose=True)

    # Train the model
    print("Starting training...")
    optimizer.fit(X, y, loss_fn)
    print("-" * 30)

    # Print final results
    print("Training complete!")
    print(f"True w: {true_w}")
    print(f"Predicted w: {optimizer.w}")
    print(f"True b: {true_b}")
    print(f"Predicted b: {optimizer.b}")
