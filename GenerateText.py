import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the Bigram Language Model
class BigramLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dropout):
        super(BigramLM, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        x = self.token_embedding_table(idx)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.fc_out(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Function to encode and decode text
def encode(s, stoi):
    return [stoi[c] for c in s]

def decode(l, itos):
    return ''.join([itos[i] for i in l])

# Hyperparameters
batch_size = 64
block_length = 32
max_iters = 10000
eval_interval = 500
learning_rate = 5e-4
embedding_dim = 256
num_heads = 8
num_layers = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 500

# Set seed for reproducibility
torch.manual_seed(1337)
np.random.seed(1337)

# Load the dataset
file_path = 'E:\\python\\Tiny Shakesphere Project\\tiny-shakespeare.txt'
with open(file_path, 'r') as file:
    text_data = file.read()

# Create character mappings
chars = sorted(list(set(text_data)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# Encode the text data
data = torch.tensor(encode(text_data, stoi), dtype=torch.long).to(device)

# Split data into training and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Function to get a batch of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_length, (batch_size,))
    x = torch.stack([data[i:i+block_length] for i in ix])
    y = torch.stack([data[i+1:i+block_length+1] for i in ix])
    return x, y

# Function to estimate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Instantiate the model
model = BigramLM(vocab_size, embedding_dim, num_heads, num_layers, dropout)
model.to(device)

# Create an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Save the trained model
torch.save(model.state_dict(), 'model_weights.pth')

# Generate 10,000 words of text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text_tensor = model.generate(context, max_new_tokens=10000)
generated_text = decode(generated_text_tensor[0].tolist(), itos)

# Save the generated text to a file
output_file_path = 'generated_text.txt'
with open(output_file_path, 'w') as f:
    f.write(generated_text)

print(f"The generated text has been saved to {output_file_path}")
