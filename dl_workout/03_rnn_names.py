"""
WORKOUT 3: RNN — Character-level Name Generation
Dataset: Handcrafted list of names (no download needed)
Goal: Train RNN to predict next character given previous characters

Architecture:
  Embedding(vocab_size, embed_dim) -> RNN(embed_dim, hidden_dim) -> Linear(hidden_dim, vocab_size)

After training: feed a start char, let it generate names character by character.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random

torch.manual_seed(42)
random.seed(42)

# ─── 1. DATA ──────────────────────────────────────────────────────────────────
names = [
    "emma", "olivia", "ava", "isabella", "sophia", "mia", "charlotte", "amelia",
    "liam", "noah", "oliver", "elijah", "william", "james", "benjamin", "lucas",
    "aria", "luna", "chloe", "penelope", "layla", "riley", "zoey", "nora",
    "ethan", "mason", "aiden", "logan", "jackson", "sebastian", "mateo", "jack",
    "harper", "lily", "ellie", "scarlett", "victoria", "madison", "grace", "ella"
]

# Build vocabulary: all unique chars + special tokens
chars = sorted(set("".join(names)))
chars = ['<SOS>', '<EOS>'] + chars  # SOS=start, EOS=end

char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}

SOS = char2idx['<SOS>']
EOS = char2idx['<EOS>']
VOCAB_SIZE = len(chars)

print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Chars: {chars}")

def name_to_tensor(name):
    """Convert name string to input/target tensors."""
    indices = [SOS] + [char2idx[c] for c in name] + [EOS]
    # input:  SOS + name chars
    # target: name chars + EOS  (shifted by 1 — predict next token)
    inp    = torch.tensor(indices[:-1], dtype=torch.long)
    target = torch.tensor(indices[1:],  dtype=torch.long)
    return inp, target

# Quick check
inp, target = name_to_tensor("emma")
print(f"\n'emma' input indices:  {inp.tolist()}")
print(f"'emma' target indices: {target.tolist()}")

# ─── 2. MODEL ─────────────────────────────────────────────────────────────────
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # lookup table
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True   # input shape: [batch, seq_len, features]
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)  # hidden state → char logits

    def forward(self, x, hidden=None):
        # x: [batch, seq_len]  (indices)
        embedded = self.embedding(x)              # → [batch, seq_len, embed_dim]
        out, hidden = self.rnn(embedded, hidden)  # out: [batch, seq_len, hidden_dim]
        logits = self.fc(out)                     # → [batch, seq_len, vocab_size]
        return logits, hidden

model = CharRNN(VOCAB_SIZE)
print(f"\nModel:\n{model}")
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

# ─── 3. LOSS + OPTIMIZER ──────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ─── 4. TRAINING LOOP ─────────────────────────────────────────────────────────
print("\n--- Training ---")
EPOCHS = 300

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    random.shuffle(names)
    for name in names:
        inp, target = name_to_tensor(name)
        inp    = inp.unsqueeze(0)    # [1, seq_len] — batch size 1
        target = target.unsqueeze(0) # [1, seq_len]

        logits, _ = model(inp)       # logits: [1, seq_len, vocab_size]

        # CrossEntropyLoss wants [N, C] and [N]
        loss = criterion(
            logits.view(-1, VOCAB_SIZE),  # [seq_len, vocab_size]
            target.view(-1)               # [seq_len]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:
        avg_loss = total_loss / len(names)
        print(f"Epoch {epoch:3d} | Avg Loss: {avg_loss:.4f}")

# ─── 5. GENERATION ────────────────────────────────────────────────────────────
def generate_name(model, start_char='a', max_len=10, temperature=0.8):
    """
    Generate a name character by character.
    temperature: lower = more confident, higher = more creative
    """
    model.eval()
    with torch.no_grad():
        # Start with SOS token
        inp = torch.tensor([[SOS]])
        hidden = None
        result = [start_char]

        # Feed the start character
        logits, hidden = model(torch.tensor([[char2idx[start_char]]]), hidden)

        for _ in range(max_len):
            # Sample next char from probability distribution
            probs = torch.softmax(logits[0, -1] / temperature, dim=0)
            next_idx = torch.multinomial(probs, 1).item()

            if next_idx == EOS:
                break

            next_char = idx2char[next_idx]
            if next_char not in ('<SOS>', '<EOS>'):
                result.append(next_char)

            logits, hidden = model(torch.tensor([[next_idx]]), hidden)

    return ''.join(result)

print("\n--- Generated Names ---")
for start in ['a', 'e', 'l', 'm', 'n', 'o', 's']:
    generated = [generate_name(model, start_char=start) for _ in range(3)]
    print(f"  '{start}...' → {generated}")

# ─── 6. UNDERSTAND HIDDEN STATE ───────────────────────────────────────────────
print("\n--- Hidden State Inspection ---")
model.eval()
with torch.no_grad():
    inp_test, _ = name_to_tensor("emma")
    inp_test = inp_test.unsqueeze(0)
    logits, hidden = model(inp_test)
    print(f"Input shape:   {inp_test.shape}")
    print(f"Logits shape:  {logits.shape}   ← [batch, seq_len, vocab_size]")
    print(f"Hidden shape:  {hidden.shape}   ← [num_layers, batch, hidden_dim]")
    print(f"Hidden carries memory across time steps!")

# ─── WHAT YOU LEARNED ─────────────────────────────────────────────────────────
print("""
KEY CONCEPTS:
  - Embedding(vocab, dim) : maps integer token → dense vector (trainable lookup)
  - RNN hidden state      : carries memory from previous time steps
  - batch_first=True      : input shape [batch, seq, features] (more intuitive)
  - Teacher forcing       : during training, feed true prev char (not predicted)
  - Temperature           : controls randomness in sampling (0.5=safe, 1.5=wild)
  - .view(-1, vocab)      : reshape for CrossEntropyLoss which needs [N, C]

  RNN LIMITATION: vanishing gradients on long sequences → use LSTM/GRU instead
  Try changing nn.RNN → nn.LSTM or nn.GRU and see if generation improves!
""")
