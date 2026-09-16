import torch


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def encode_text(text, tokenizer):
    return torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long
    )


def split_data(data, train_ratio=0.9):
    split_index = int(train_ratio * len(data))
    return data[:split_index], data[split_index:]
