import torch
import numpy as np

from utils import get_tokens_paths


class DataLoader:
    def __init__(self, B: int, T: int, split: str, tokens_dir: str, loop: bool) -> None:
        print(f"create dataloader: {split}")
        self.tokens = None
        self.current_position = None
        self.current_file_index = None
        self.data_files = get_tokens_paths(tokens_dir)
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
        print(filename)
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.current_position + self.B * self.T >= len(self.tokens):
            self.current_file_index += 1
            if self.current_file_index >= len(self.data_files):
                # self.reset()  # Reset au premier fichier si tous ont été traités
                pass
            else:
                self.tokens = self.load_tokens(self.data_files[self.current_file_index])
                self.current_position = 0

        end_pos = self.current_position + self.B * self.T + 1
        if end_pos < len(self.tokens):  # Pas assez de données pour le dernier lot dans le fichier courant
            x = self.tokens[self.current_position:self.current_position + self.B * self.T]
            y = self.tokens[self.current_position + 1: end_pos]

            x = x.view(self.B, self.T)
            y = y.view(self.B, self.T)

            self.current_position += self.B * self.T
            return x, y

    def __bool__(self) -> bool:
        return (self.current_file_index < len(self.data_files)) or (
                self.current_position + self.B * self.T + 1 < len(self.tokens))
