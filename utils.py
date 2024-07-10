from tokenizers import Tokenizer
import os
import json
from torch import nn

from config import *


def get_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(os.path.join(DATA_DIR,VOCAB_PATH))


def get_tokens_paths(tokens_dir: str, split: str) -> list[str]:
    return sorted([os.path.join(tokens_dir, f) for f in (os.listdir(tokens_dir)) if f.endswith(".npy") and split in f])


def get_vocab() -> dict:
    with open(VOCAB_PATH, "r") as f:
        data = json.load(f)
    return data["model"]["vocab"]


def save_model(model: nn.Module, name: str) -> None:
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, name))


def load_weights(model: nn.Module, name: str) -> nn.Module:
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    return model
