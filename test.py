from Dataloader import DataLoader
from config import *

dl = DataLoader(batch_size, block_size, "", TOKENS_DIR)

while dl:
    data = dl.next_batch()
    if data is not None:
        x, y = data
        print(x.shape)
