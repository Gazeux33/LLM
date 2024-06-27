import tiktoken
import torch
import numpy as np

from utils import get_tokens_paths


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('data/tinyshakespeare.txt', "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        self.current_position = 0

        print(f"for 1 epochs:{len(self.tokens)}    {len(self.tokens) // (B * T)} batches")

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets

        self.current_position += B * T

        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

    class DataLoader:
        def __init__(self, B: int, T: int, split: str):
            self.tokens = None
            self.current_position = None
            self.current_file_index = None
            self.data_files = get_tokens_paths()
            self.B = B
            self.T = T
            self.split = split
            self.reset()

        def reset(self) -> None:
            self.current_file_index = 0
            self.tokens = self.load_tokens(self.data_files[self.current_file_index])
            self.current_position = 0

        @staticmethod
        def load_tokens(filename: str) -> torch.Tensor:
            npt = np.load(filename)
            npt = npt.astype(np.int32)
            ptt = torch.tensor(npt, dtype=torch.long)
            return ptt

        def next_batch(self) -> torch.Tensor:
            if self.current_position + self.B * self.T >= len(self.tokens):
                self.current_file_index += 1
                if self.current_file_index >= len(self.data_files):
                    raise StopIteration("everything has been processed")
                self.tokens = self.load_tokens(self.data_files[self.current_file_index])
                self.current_position = 0

            end_pos = self.current_position + self.B * self.T + 1
            if end_pos > len(self.tokens):
                raise StopIteration("not enought data for the last batch")

            x = self.tokens[self.current_position: self.current_position + self.B * self.T]
            y = self.tokens[self.current_position + 1: end_pos]

            # Redimensionner x et y
            x = x.view(self.B, self.T)
            y = y.view(self.B, self.T)

            self.current_position += self.B * self.T
            return x, y
