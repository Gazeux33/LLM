from Dataloader import DataLoader
import numpy as np

B,T,split = 8,16,""


dl = DataLoader(B,T,split)

#%%
for i in range(10_000_000):
    x,y = dl.next_batch()