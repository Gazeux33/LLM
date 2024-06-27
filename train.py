import torch

from config import *
from model import GPT
from Dataloader import DataLoader

print(VOCAB_PATH)
dataloader = DataLoader(batch_size, block_size, "", TOKENS_DIRECTORY)
model = GPT()
model.to(DEVICE)

x, y = dataloader.next_batch()
x, y = x.to(DEVICE), y.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

iter = 50

for i in range(iter):
    optimizer.zero_grad()
    out, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(loss.item())



