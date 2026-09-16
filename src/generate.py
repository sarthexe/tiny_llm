import torch
import torch.nn.functional as F

from model import TinyLLM


def generate(
    model,
    context,
    max_new_tokens,
    block_size,
    temperature=0.8,
    top_k=None
):
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # only give the model the latest context window
            context_for_model = context[:, -block_size:]

            logits = model(context_for_model)

            # only use the prediction from the last position
            logits = logits[:, -1, :]

            # control randomness
            logits = logits / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            # convert logits into probabilities
            probs = F.softmax(logits, dim=-1)

            # randomly sample the next token
            next_token = torch.multinomial(probs, num_samples=1)

            # add it to the context
            context = torch.cat((context, next_token), dim=1)

    return context


def load_checkpoint(path, vocab_size, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]

    model = TinyLLM(
        vocab_size=vocab_size,
        n_embd=config["n_embd"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        block_size=config["block_size"],
        dropout=config.get("dropout", 0.2)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model
