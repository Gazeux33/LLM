from tokenizers import Tokenizer
import os

from config import *


def get_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(VOCAB_PATH)


def get_tokens_paths() -> list[str]:
    return sorted([f for f in (os.listdir(TOKENS_DIRECTORY)) if f.endswith(".npy")])
