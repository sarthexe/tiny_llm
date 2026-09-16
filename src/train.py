import torch
import torch.nn.functional as F

from model import TinyLLM


def get_batch(data, batch_size, block_size, device):
    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[start:start + block_size]
        for start in starts
    ])

    y = torch.stack([
        data[start + 1:start + block_size + 1]
        for start in starts
    ])

    return x.to(device), y.to(device)


def estimate_loss(
    model,
    train_data,
    val_data,
    batch_size,
    block_size,
    vocab_size,
    device,
    eval_iters=20
):
    losses = {}
    model.eval()

    with torch.no_grad():
        for split, data in [("train", train_data), ("val", val_data)]:
            split_losses = []

            for _ in range(eval_iters):
                x, y = get_batch(data, batch_size, block_size, device)
                logits = model(x)

                B, T = x.shape
                logits = logits.reshape(B * T, vocab_size)
                targets = y.reshape(B * T)

                loss = F.cross_entropy(logits, targets)
                split_losses.append(loss.item())

            losses[split] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


def train_model(
    train_data,
    val_data,
    vocab_size,
    block_size=128,
    batch_size=32,
    n_embd=128,
    num_heads=4,
    num_layers=4,
    learning_rate=0.001,
    dropout=0.2,
    num_steps=5001,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyLLM(
        vocab_size=vocab_size,
        n_embd=n_embd,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    print(f"Device: {device}")
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )

    model.train()

    for step in range(num_steps):
        x, y = get_batch(
            train_data,
            batch_size,
            block_size,
            device
        )

        optimizer.zero_grad()

        logits = model(x)

        B, T = x.shape
        logits = logits.reshape(B * T, vocab_size)
        targets = y.reshape(B * T)

        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                batch_size,
                block_size,
                vocab_size,
                device
            )

            print(
                f"Step {step} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f}"
            )

    return model


def save_checkpoint(model, optimizer, path, config):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "config": config,
        },
        path
    )


if __name__ == "__main__":
    print("Import train_model() and provide tokenized train/validation data.")
