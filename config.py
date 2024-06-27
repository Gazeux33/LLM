import torch
from utils import *

VOCAB_PATH = "data/vocab.json"
ENCODING = "utf-8"
TOKENS_DIRECTORY = "data/tokens"

device_name = "cpu"
if torch.cuda.is_available():
    device_name = "cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # device_name = "mps"
DEVICE = torch.device(device_name)
print(f"using device: {DEVICE}")

vocab_size = 30_000

block_size = 8
batch_size = 4
lr = 6e-4

n_block = 12
n_head = 12
n_embd = 768
