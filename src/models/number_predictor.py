"""
🧠 Neuro Brain — Number Predictor
Feedforward Neural Network (FNN) for numeric prediction tasks.

Architecture Overview:
    Input(n features) → Dense(128)+ReLU → Dense(64)+ReLU → Dense(32)+ReLU → Dense(1)+Linear

Activation Choices:
    • Hidden layers: ReLU — adds non-linearity so the network can learn
      curved relationships (e.g., diminishing returns of square footage on price).
      ReLU is fast, avoids vanishing gradients, and works well for regression.
    • Output layer: Linear (no activation) — the prediction is an unbounded
      real number (price, temperature, stock value), so we don't clamp it.
"""

import numpy as np
# Deferred imports inside classes to prevent dependency crashes


class NumberPredictor:
    """
    Feedforward Neural Network for predicting continuous numbers.

    Architecture:
        ┌─────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐
        │  INPUT  │───→│ HIDDEN 1  │───→│ HIDDEN 2 │───→│ HIDDEN 3 │───→│ OUTPUT │
        │ n feats │    │ 128+ReLU  │    │ 64+ReLU  │    │ 32+ReLU  │    │ 1 unit │
        │         │    │ +Dropout  │    │ +Dropout │    │ +Dropout │    │ Linear │
        └─────────┘    └───────────┘    └──────────┘    └──────────┘    └────────┘

    Why this design:
        - 3 hidden layers: enough depth to learn complex feature interactions
        - Decreasing width (128→64→32): forces the network to compress information
          into the most important patterns — like a funnel
        - Dropout(0.2): randomly disables 20% of neurons during training to prevent
          the network from memorizing (overfitting) instead of learning
    """

    def __init__(self, input_dim=1, hidden_layers=None, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [128, 64, 32]
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.history = None

    def _build_model(self):
        """
        Build the FNN layer by layer.

        Layer-by-layer explanation:
        ─────────────────────────────────────────────────────────
        1. Input Shape: (n_features,)
           - Tells the network how many input features to expect.
           - Example: 4 features = [square_feet, bedrooms, age, distance]

        2. Dense(128, activation='relu')  ← HIDDEN LAYER 1
           - 128 neurons, each connected to all inputs.
           - ReLU activation: f(x) = max(0, x)
           - WHY ReLU: Adds non-linearity. Without it, stacking layers
             would be equivalent to a single linear equation (useless).
             ReLU is fast to compute and avoids the vanishing gradient
             problem that plagues sigmoid in deep networks.

        3. Dropout(0.2)
           - Randomly sets 20% of neuron outputs to 0 during training.
           - WHY: Forces the network to not rely on any single neuron,
             creating redundant pathways. This prevents overfitting.

        4. Dense(64, activation='relu')  ← HIDDEN LAYER 2
           - Fewer neurons = compresses learned features.
           - The network is forced to identify the MOST important patterns.

        5. Dense(32, activation='relu')  ← HIDDEN LAYER 3
           - Further compression. By now, the network has distilled
             raw inputs into abstract features highly relevant to the target.

        6. Dense(1, activation=None)  ← OUTPUT LAYER
           - Single neuron outputs the prediction.
           - NO activation (linear): the output is a raw number.
           - WHY linear: We're predicting a continuous value (like price).
             Sigmoid would clamp it to [0,1]. Softmax is for classification.
             Linear lets the output be any real number: -∞ to +∞.
        ─────────────────────────────────────────────────────────
        """
        model = keras.Sequential(name="NeuroBrain_NumberPredictor")

        # ── Layer 1: Input + First Hidden Layer ───────────────
        # Dense = fully connected layer (every input connects to every neuron)
        # activation='relu' = Rectified Linear Unit: max(0, x)
        # WHY ReLU: Fast, no vanishing gradient, learns non-linear patterns
        model.add(keras.layers.Dense(
            self.hidden_layers[0],
            activation='relu',
            input_shape=(self.input_dim,),
            name='hidden_1_relu'
        ))
        model.add(keras.layers.Dropout(0.2, name='dropout_1'))

        # ── Layer 2+: Additional Hidden Layers ────────────────
        for i, units in enumerate(self.hidden_layers[1:], start=2):
            # Each subsequent layer has fewer neurons (funnel shape)
            # This compresses features into essential patterns
            model.add(keras.layers.Dense(
                units,
                activation='relu',         # ReLU: consistent activation
                name=f'hidden_{i}_relu'
            ))
            model.add(keras.layers.Dropout(0.2, name=f'dropout_{i}'))

        # ── Output Layer: Linear Activation ───────────────────
        # activation=None means LINEAR: f(x) = x (pass-through)
        # WHY: Regression output should be unbounded (any real number)
        # Using sigmoid here would crush output to [0,1] — BAD for prices!
        model.add(keras.layers.Dense(
            1,
            activation=None,               # Linear — raw prediction value
            name='output_linear'
        ))

        # ── Compile with Adam optimizer + MSE loss ────────────
        # Adam: Adaptive learning rate optimizer (best general-purpose choice)
        # MSE: Mean Squared Error — penalizes large prediction errors heavily
        # MAE: Mean Absolute Error — for human-readable error monitoring
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model on data."""
        print(f"🧠 Training Number Predictor for {epochs} epochs...")
        print(f"   Architecture: Input({X.shape[1]}) → " +
              " → ".join(f"Dense({h})+ReLU" for h in self.hidden_layers) +
              " → Dense(1)+Linear")
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return self.history

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X, verbose=0)

    def plot_training(self):
        """Visualize training progress."""
        if self.history is None:
            print("⚠️ No training history. Train the model first.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Val Loss')
        ax1.set_title('Loss (MSE) Over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.history.history['mae'], label='Train MAE')
        ax2.plot(self.history.history['val_mae'], label='Val MAE')
        ax2.set_title('Error (MAE) Over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('data/models/number_predictor_training.png', dpi=150)
        plt.show()
        print("📊 Training plot saved to data/models/number_predictor_training.png")

    def save(self, path="data/models/number_predictor.keras"):
        """Save the trained model."""
        self.model.save(path)
        print(f"💾 Model saved to {path}")

    def load(self, path="data/models/number_predictor.keras"):
        """Load a saved model."""
        self.model = keras.models.load_model(path)
        print(f"📂 Model loaded from {path}")

    def summary(self):
        """Print model architecture."""
        self.model.summary()

    def explain_architecture(self):
        """Print a human-readable explanation of the architecture."""
        print(f"""
╔══════════════════════════════════════════════════════════╗
║           🧠 NUMBER PREDICTOR — ARCHITECTURE            ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  INPUT ({self.input_dim} features)                              ║
║    │                                                     ║
║    ▼                                                     ║
║  ┌─────────────────────────────────────────────────┐     ║
║  │ HIDDEN LAYER 1: Dense({self.hidden_layers[0]}) + ReLU          │     ║
║  │ Activation: ReLU = max(0, x)                    │     ║
║  │ Purpose: Learn non-linear feature combinations  │     ║
║  │ + Dropout(0.2): prevent overfitting             │     ║
║  └─────────────────────────┬───────────────────────┘     ║
║                            │                             ║
║                            ▼                             ║""")
        for i, units in enumerate(self.hidden_layers[1:], start=2):
            print(f"""║  ┌─────────────────────────────────────────────────┐     ║
║  │ HIDDEN LAYER {i}: Dense({units}) + ReLU             │     ║
║  │ Activation: ReLU — compress features further    │     ║
║  │ + Dropout(0.2)                                  │     ║
║  └─────────────────────────┬───────────────────────┘     ║
║                            │                             ║
║                            ▼                             ║""")
        print(f"""║  ┌─────────────────────────────────────────────────┐     ║
║  │ OUTPUT: Dense(1) + Linear (no activation)       │     ║
║  │ Activation: None — unbounded real number output │     ║
║  │ Purpose: Predict a continuous value             │     ║
║  └─────────────────────────────────────────────────┘     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ── Quick Demo ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  🧠 Number Predictor — Full Demo")
    print("=" * 55)

    # Generate sample data: house price prediction
    np.random.seed(42)
    n = 1000
    square_feet = np.random.randint(500, 5000, n).astype(float)
    bedrooms = np.random.randint(1, 6, n).astype(float)
    age = np.random.randint(0, 50, n).astype(float)
    distance = np.random.uniform(0.5, 30, n)

    X = np.column_stack([square_feet, bedrooms, age, distance])
    y = (square_feet * 150 + bedrooms * 20000 - age * 1000 -
         distance * 5000 + np.random.randn(n) * 10000)

    # Normalize (important for neural networks!)
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # Create, explain, and train
    predictor = NumberPredictor(input_dim=4, hidden_layers=[128, 64, 32])
    predictor.explain_architecture()
    predictor.summary()
    predictor.train(X_train, y_train, epochs=50)

    # Evaluate
    test_predictions = predictor.predict(X_test)
    test_predictions_actual = scaler_y.inverse_transform(test_predictions)
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    print("\n📊 Sample Predictions (first 5):")
    print(f"  {'Predicted':>12}  {'Actual':>12}  {'Error':>10}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*10}")
    for pred, actual in zip(test_predictions_actual[:5], y_test_actual[:5]):
        error = abs(pred[0] - actual[0])
        print(f"  ${pred[0]:>10,.0f}  ${actual[0]:>10,.0f}  ${error:>8,.0f}")
