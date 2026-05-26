"""
WORKOUT 1: ANN — Binary Classification
Dataset: Pima Indians Diabetes (sklearn style)
Goal: Predict if a person has diabetes (0 or 1)

Architecture:
  Input(8) -> Linear(16) -> ReLU -> Linear(8) -> ReLU -> Linear(1) -> Sigmoid
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

torch.manual_seed(42)

# ─── 1. DATA ──────────────────────────────────────────────────────────────────
# Simulating diabetes-like binary classification with 8 features
X, y = make_classification(
    n_samples=768, n_features=8, n_informative=5,
    n_redundant=2, random_state=42
)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # shape: [N, 1]
y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ─── 2. MODEL ─────────────────────────────────────────────────────────────────
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),   # 8 input features → 16 neurons
            nn.ReLU(),
            nn.Linear(16, 8),   # 16 → 8 neurons
            nn.ReLU(),
            nn.Linear(8, 1),    # 8 → 1 output
            nn.Sigmoid()        # output between 0 and 1 (probability)
        )

    def forward(self, x):
        return self.net(x)

model = ANN()
print(f"\nModel:\n{model}")
print(f"Total params: {sum(p.numel() for p in model.parameters())}")

# ─── 3. LOSS + OPTIMIZER ──────────────────────────────────────────────────────
criterion = nn.BCELoss()             # Binary Cross Entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ─── 4. TRAINING LOOP ─────────────────────────────────────────────────────────
print("\n--- Training ---")
EPOCHS = 50

for epoch in range(1, EPOCHS + 1):
    model.train()

    # Forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # Backward pass
    optimizer.zero_grad()   # clear old gradients
    loss.backward()         # compute gradients
    optimizer.step()        # update weights

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test)
            accuracy = ((test_pred > 0.5).float() == y_test).float().mean()
        print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f} | Acc: {accuracy.item()*100:.1f}%")

# ─── 5. FINAL EVALUATION ──────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    preds = model(X_test)
    preds_class = (preds > 0.5).float()
    final_acc = (preds_class == y_test).float().mean()
    print(f"\nFinal Test Accuracy: {final_acc.item()*100:.2f}%")

# ─── WHAT YOU LEARNED ─────────────────────────────────────────────────────────
print("""
KEY CONCEPTS:
  - nn.Linear(in, out)  : fully connected layer (weights + bias)
  - ReLU                : kills negatives, keeps positives (non-linearity)
  - Sigmoid             : squashes output to [0,1] for binary probability
  - BCELoss             : penalizes wrong probability predictions
  - optimizer.zero_grad(): MUST clear grads before each backward()
  - loss.backward()     : computes all gradients via chain rule (autograd)
  - optimizer.step()    : w = w - lr * grad
""")
