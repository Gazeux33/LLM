import torch

# *** PATHS ***
VOCAB_PATH = "data/vocab.json"
ENCODING = "utf-8"
TOKENS_DIR = "data/tokens"
MODEL_DIR = "models"

# *** DEVICE ***
device_name = "cpu"
if torch.cuda.is_available():
    device_name = "cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
# device_name = "mps"
DEVICE = torch.device(device_name)
print(f"using device: {DEVICE}")

# *** TRAIN ***
block_size = 8
batch_size = 4
lr = 6e-4
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.1
EPOCHS = 2
EVAL_FREQ = 50
SAVE_FREQ = 100

# *** MODEL ***
VOCAB_SIZE = 30_000
N_BLOCK = 12
N_HEAD = 12
N_EMBD = 768

# *** OTHER ***
torch.set_float32_matmul_precision('high')
