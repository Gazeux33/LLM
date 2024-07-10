import json
import math
import os.path
from typing import Tuple
import time

import utils
from config import *
from model import GPT
from Dataloader import DataLoader
from optimizer import AdamW


# nvidia-smi
# TO ADD : gradient accumulation, metrcis plus viables
class Train:
    def __init__(self, compile_model: bool = False, load: bool = True):
        self.train_dataloader = DataLoader(batch_size, block_size, "train", TOKENS_DIR, loop=False)
        self.test_dataloader = DataLoader(batch_size, block_size, "test", TOKENS_DIR, loop=True)
        self.metrics = Metrics()
        self.model = GPT()
        self.model.to(DEVICE)
        self.step = 0
        self.prev_nb_step = self.train_dataloader.count_total_batches()
        print(f"nb_batchs:{self.prev_nb_step}")
        self.warmup_steps = 0.1 * self.prev_nb_step

        if load:
            print("loading model....")
            self.model = utils.load_weights(self.model, "last.pth")
            self.metrics.load_metrics()
            self.step = self.metrics.current_state["step"]
            self.tokens_buffer()

        if compile_model:
            print("compile model....")
            self.model = torch.compile(self.model)

        self.optimizer = AdamW(self.model.parameters(), learning_rate=max_lr)

        assert total_batch_size % (batch_size * block_size) == 0
        self.gradient_accumulation_steps = total_batch_size // (batch_size * block_size)
        print(f"gradient_accumulation_steps:{self.gradient_accumulation_steps}")

    def train(self) -> None:
        for i in range(EPOCHS):
            while self.train_dataloader:
                start = time.time()
                self.model.train()

                current_lr = self.get_lr(self.step)
                for g in self.optimizer.param_groups:
                    g['lr'] = current_lr

                self.optimizer.zero_grad()
                for grad_step in range(self.gradient_accumulation_steps):
                    x, y = self.train_dataloader.next_batch()
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    logits, loss = self.predict(x, y)
                    loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.step += 1
                # print(loss.item())
                self.metrics.update_metrics({'train_loss': round(loss.item(), 2),
                                             "step": self.step,
                                             "tokens": self.metrics.current_state["tokens"] + (
                                                     block_size * batch_size * self.gradient_accumulation_steps),
                                             "epochs": i + 1},

                                            )
                elapsed_time = time.time() - start
                self.update()
                self.print_state(gap=round(elapsed_time * 1000, 2))

    def print_state(self, gap: float) -> None:
        m = ""
        for k, v in self.metrics.current_state.items():
            m += f"{k}:{v}  "
        # m += f"lr:{self.optimizer.param_groups[0]['lr']}  "
        m += f"time:{gap} "
        print(m)

    def predict(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if device_name == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, loss = self.model(x, y)
        else:
            logits, loss = self.model(x, y)
        return logits, loss

    def update(self) -> None:
        if self.step % EVAL_FREQ == 0:
            self.eval()
        if self.step % SAVE_FREQ == 0:
            utils.save_model(self.model, "last.pth")
            print("saving model...")
            self.metrics.save_metrcis()

    def eval(self) -> None:
        print("evaluating...")
        self.model.eval()
        x_test, y_test = self.test_dataloader.next_batch()
        x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)
        logits, loss_test = self.model(x_test, y_test)
        acc_test = self.model.accuracy(logits, y_test)
        self.metrics.update_metrics(
            {'test_loss': round(loss_test.item(), 2), 'test_accuracy': round(acc_test.item(), 2)})

    def get_lr(self, it: int):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_steps:
            return max_lr * (it + 1) / self.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.prev_nb_step:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_steps) / (self.prev_nb_step - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    def tokens_buffer(self):
        buffer = 0
        while self.metrics.current_state["tokens"] > buffer:
            x, y = self.train_dataloader.next_batch()
            buffer += x.size(0) * x.size(1)
        print(f"buffered:{buffer}")


class Metrics:
    def __init__(self):

        self.current_state = {
            'step': 0,
            'tokens': 0,
            'epochs': 0,
            'train_loss': 0,
            'test_loss': 0,
            'test_accuracy': 0
        }

        self.metrics_history = {
            'train_loss': [],
            'test_loss': [],
            'test_accuracy': []
        }

    def update_metrics(self, metrics: dict) -> None:
        for k, v in metrics.items():
            if k in self.current_state:
                self.current_state[k] = v
            if k in self.metrics_history:
                self.metrics_history[k].append(v)

    def save_metrcis(self) -> None:
        data = {"current_state": self.current_state, "history": self.metrics_history}
        with open(os.path.join(MODEL_DIR, "metrics.json"), "w+") as f:
            json.dump(data, f, indent=2)

    def load_metrics(self) -> None:
        path = os.path.join(MODEL_DIR, "metrics.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                print(data)
            self.current_state = data["current_state"]
            self.metrics_history = data["history"]


if __name__ == "__main__":
    trainer = Train(compile_model=True, load=False)
    trainer.train()
