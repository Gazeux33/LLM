from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

data_files = ["data/cleaned_data.txt"]
out_path = "data/vocab.json"

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

if __name__ == "__main__":
    tokenizer.train(data_files, trainer)
    tokenizer.save(out_path)