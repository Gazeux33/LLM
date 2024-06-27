from tokenizers import Tokenizer
import numpy as np
import os
import time

from config import *
from utils import get_tokenizer

data_path = "data/cleaned_data.txt"
token_per_file = 25_000_000


def split_encoded_text(src_path: str, tokenizer: Tokenizer, tok_per_file: int) -> None:
    tokens_accumulated = []
    file_index = 0

    with open(src_path, "r", encoding=ENCODING) as src_file:
        for line in src_file:
            encoded = tokenizer.encode(line.strip())
            tokens_accumulated.extend(encoded.ids)

            if len(tokens_accumulated) >= tok_per_file:
                tokens_to_save = tokens_accumulated[:tok_per_file]
                save_npy_file(os.path.join(TOKENS_DIRECTORY, f"tokens_{file_index}.npy"), np.array(tokens_to_save))
                tokens_accumulated = tokens_accumulated[tok_per_file:]
                file_index += 1

    if tokens_accumulated:
        save_npy_file(os.path.join(TOKENS_DIRECTORY, f"tokens_{file_index}.npy"), np.array(tokens_accumulated))


def save_npy_file(path: str, data) -> None:
    np.save(path, data)
    print(f"saved:{path}")


if __name__ == "__main__":
    b = time.time()
    print("launch tokenizations.....")
    tok = get_tokenizer()
    split_encoded_text(data_path, tok, token_per_file)
    e = time.time()
    print("ended.")
    print(f"time:{e-b} s")
