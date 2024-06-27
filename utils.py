from tokenizers import Tokenizer
import os
import json

from config import *


def get_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(VOCAB_PATH)


def get_tokens_paths(tokens_dir:str) -> list[str]:
    return sorted([os.path.join(tokens_dir, f) for f in (os.listdir(tokens_dir)) if f.endswith(".npy")])


def get_vocab() -> dict:
    with open(VOCAB_PATH, "r") as f:
        data = json.load(f)
    return data["model"]["vocab"]


