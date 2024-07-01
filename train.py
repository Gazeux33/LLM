import json
import os.path

import utils
from config import *
from model import GPT
from Dataloader import DataLoader
from optimizer import AdamW


# TO ADD : gradient accumulation, metrcis plus viables
class Train:
    def __init__(self, compile_model: bool = False, load: bool = True):
        self.train_dataloader = DataLoader(batch_size, block_size, "train", TOKENS_DIR, loop=False)
        self.test_dataloader = DataLoader(batch_size, block_size, "test", TOKENS_DIR, loop=True)
        self.metrics = Metrics()
        self.model = GPT()
        self.model.to(DEVICE)
        self.step = 0

        if load:
            print("loading model....")
            self.model = utils.load_weights(self.model, "last.pth")
            self.metrics.load_metrics()
        if compile_model:
            print("compile model....")
            self.model = torch.compile(self.model)

        self.optimizer = AdamW(self.model.parameters(), learning_rate=lr)

    def train(self) -> None:
        for i in range(EPOCHS):
            while self.train_dataloader:
                self.model.train()
                x, y = self.train_dataloader.next_batch()
                x, y = x.to(DEVICE), y.to(DEVICE)
                self.optimizer.zero_grad()
                logits, loss = self.predict(x, y)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.step += 1
                # print(loss.item())
                self.metrics.update_metrics({'train_loss': round(loss.item(), 2),
                                             "step": self.metrics.current_state["step"] + 1,
                                             "tokens": self.metrics.current_state["tokens"] + (
                                                     block_size * batch_size),
                                             "epochs": i + 1},

                                            )
                self.update()
                self.print_state()

    def print_state(self) -> None:
        m = ""
        for k, v in self.metrics.current_state.items():
            m += f"{k}:{v}  "
        print(m)

    def predict(self, x: torch.Tensor, y: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        if device_name == "cuda":
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
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
        print("EVAL !!!!")
        self.model.eval()
        x_test, y_test = self.test_dataloader.next_batch()
        x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)
        logits, loss_test = self.model(x_test, y_test)
        acc_test = self.model.accuracy(logits, y_test)
        self.metrics.update_metrics(
            {'test_loss': round(loss_test.item(), 2), 'test_accuracy': round(acc_test.item(), 2)})


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
    trainer = Train(compile_model=False, load=True)
    trainer.train()
