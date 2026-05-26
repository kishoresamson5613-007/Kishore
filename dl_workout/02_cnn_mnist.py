"""
WORKOUT 2: CNN — Image Classification
Dataset: MNIST handwritten digits (0-9)
Goal: Classify 28x28 grayscale images into 10 digit classes

Architecture:
  Conv(1→16,3x3) -> ReLU -> MaxPool(2x2)
  Conv(16→32,3x3) -> ReLU -> MaxPool(2x2)
  Flatten -> Linear(32*5*5, 128) -> ReLU -> Linear(128, 10)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)

# ─── 1. DATA ──────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),           # PIL image → tensor [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])

train_data = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
print(f"Sample image shape: {train_data[0][0].shape}")  # [1, 28, 28]

# ─── 2. MODEL ─────────────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extractor — learns spatial patterns
        self.conv_block = nn.Sequential(
            # Input: [batch, 1, 28, 28]
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # → [batch, 16, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → [batch, 16, 14, 14]

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # → [batch, 32, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → [batch, 32, 7, 7]
        )

        # Classifier — maps features to class scores
        self.classifier = nn.Sequential(
            nn.Flatten(),            # [batch, 32, 7, 7] → [batch, 1568]
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)       # 10 digit classes
            # No Softmax — CrossEntropyLoss includes it internally
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x

model = CNN()
print(f"\nModel:\n{model}")
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

# ─── 3. LOSS + OPTIMIZER ──────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()   # multi-class: applies softmax + NLLLoss internally
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"\nUsing device: {device}")

# ─── 4. TRAINING LOOP ─────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)           # forward
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

print("\n--- Training ---")
EPOCHS = 3  # 3 epochs is enough for MNIST ~98%

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    test_loss,  test_acc  = evaluate(model, test_loader, criterion)
    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Test Loss: {test_loss:.4f} Acc: {test_acc*100:.1f}%")

# ─── 5. INSPECT A PREDICTION ──────────────────────────────────────────────────
model.eval()
sample_img, sample_label = test_data[0]
with torch.no_grad():
    logits = model(sample_img.unsqueeze(0).to(device))  # add batch dim
    pred = logits.argmax(1).item()
print(f"\nSample — True: {sample_label}, Predicted: {pred}")

# ─── WHAT YOU LEARNED ─────────────────────────────────────────────────────────
print("""
KEY CONCEPTS:
  - Conv2d(in_ch, out_ch, k): slides k×k filter, learns local patterns
  - MaxPool2d(2)            : halves spatial size, keeps strongest signal
  - Flatten()               : collapses [C, H, W] into a 1D vector for Linear
  - CrossEntropyLoss        : combines log_softmax + NLLLoss (don't add Softmax)
  - argmax(1)               : picks the class with highest logit score
  - DataLoader              : batches + shuffles data automatically
  - .to(device)             : moves tensor/model to GPU if available
""")
