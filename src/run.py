import torch

from .data import encode_text, load_text, split_data
from .generate import generate
from .tokenizer import ByteBPETokenizer
from .train import train_model


DATA_PATH = "input.txt"
NUM_MERGES = 500

BLOCK_SIZE = 128
BATCH_SIZE = 32
N_EMBD = 128
NUM_HEADS = 4
NUM_LAYERS = 4
LEARNING_RATE = 0.001
DROPOUT = 0.2
NUM_STEPS = 5001


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_text(DATA_PATH)

    print("Training tokenizer...")
    tokenizer = ByteBPETokenizer().train(text[:100_000], NUM_MERGES)

    data = encode_text(text, tokenizer)
    train_data, val_data = split_data(data)

    vocab_size = len(tokenizer.itos)

    model, optimizer = train_model(
        train_data=train_data,
        val_data=val_data,
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        n_embd=N_EMBD,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        learning_rate=LEARNING_RATE,
        dropout=DROPOUT,
        num_steps=NUM_STEPS,
        device=device
    )

    context = torch.tensor(
        [[tokenizer.encode("T")[0]]],
        dtype=torch.long,
        device=device
    )

    output = generate(
        model,
        context,
        max_new_tokens=500,
        block_size=BLOCK_SIZE,
        temperature=0.8,
        top_k=50
    )

    print("\nGenerated text:\n")
    print(tokenizer.decode(output[0].tolist()))

    config = {
        "n_embd": N_EMBD,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "block_size": BLOCK_SIZE,
        "dropout": DROPOUT,
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        "tiny_llm_bpe.pth"
    )


if __name__ == "__main__":
    main()
