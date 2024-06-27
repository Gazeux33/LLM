import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
#elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#device = "mps"
print(f"using device:{device}")

DATA_PATH = "../data/tinyshakespeare.txt"
iter = 5000
block_size = 256
batch_size = 64
n_embedding = 384
lr = 3e-4
n_block = 6
n_head = 6
dropout = 0.2

with (open(DATA_PATH, "r", encoding="utf-8") as f):
    text = f.read()
len(text)

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [char_to_int[e] for e in s]
decode = lambda i: "".join([int_to_char[e] for e in i])

encode_data = encode(text)
data = torch.tensor(encode_data, dtype=torch.long)

train_size = 0.9
n = int(len(data) * train_size)
train_data = data[:n]
test_data = data[n:]


def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


xb, yb = get_batch(train_data)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # residual connections
        x = x + self.sa(self.layer_norm1(x))
        x = x + self.ffwd(self.layer_norm2(x))
        return x


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key_layer = nn.Linear(n_embedding, head_size, bias=False)
        self.query_layer = nn.Linear(n_embedding, head_size, bias=False)
        self.value_layer = nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key_layer(x)
        q = self.query_layer(x)
        v = self.value_layer(x)

        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embedding, n_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_emb):
        # feed forward multiplied by 4
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, n_emb * 4),
            nn.ReLU(),
            nn.Linear(n_emb * 4, n_emb),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding = nn.Embedding(block_size, n_embedding)

        self.blocks = nn.Sequential(*[Block(n_embedding, n_head=n_head) for _ in range(n_block)])
        self.ln = nn.LayerNorm(n_embedding)
        self.linear = nn.Linear(n_embedding, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx , targets are (B,T)
        tok_emb = self.token_embedding(idx)  # (batch,token,channels)
        pos_emb = self.position_embedding(torch.arange(T, device=device))  # (T,C)

        x = tok_emb + pos_emb  # (B,T,C) =  broadcasting
        x = self.blocks(x)
        x = self.ln(x)
        out = self.linear(x)  # (b,t,vocab size )

        if targets is None:
            loss = None
        else:
            B, T, C = out.shape
            out = out.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(out, targets)
        return out, loss

    def generate(self, idx, max_new_token):
        for _ in range(max_new_token):
            idx_cond = idx[:, -block_size:]
            logits, l = self(idx_cond)
            logits = logits[:, -1, :]  # => (B,C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx


m = BigramLanguageModel()
m.to(device)
opt = torch.optim.AdamW(m.parameters(), lr=lr)

for step in tqdm(range(iter)):
    xb, yb = get_batch(train_data)
    logits, loss = m(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
print(loss.item())

idx = torch.zeros((1, 1), dtype=torch.long)
result = decode(m.generate(idx, max_new_token=300)[0].tolist())
print(result)
