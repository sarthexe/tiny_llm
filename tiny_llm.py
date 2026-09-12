import torch
import torch.nn as nn
import torch.nn.functional as F

text = """
the cat sat on the mat.
the dog sat on the floor.
the cat likes the dog.
"""

# make the vocabulary
chars = sorted(list(set(text)))

print("Vocabulary:", chars)
print("Vocabulary size:", len(chars))

vocab_size = len(chars)


# character <-> number mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# convert the text into token IDs
encoded = [stoi[ch] for ch in text]

data = torch.tensor(encoded, dtype=torch.long)


# model settings
block_size = 8
batch_size = 4

n_embd = 16
num_heads = 4

learning_rate = 0.001
num_steps = 300


# one attention head
class Head(nn.Module):

    def __init__(self, head_size, n_embd):
        super().__init__()

        self.head_size = head_size

        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        # prevents tokens from looking into the future
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(block_size, block_size)
            )
        )


    def forward(self, x):

        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # calculate attention scores
        attention_score = (
            q @ k.transpose(-2, -1)
        ) / (self.head_size ** 0.5)

        # hide future tokens
        masked_attention_score = attention_score.masked_fill(
            self.causal_mask[:T, :T] == 0,
            float("-inf")
        )

        # convert scores into probabilities
        attention_weights = F.softmax(
            masked_attention_score,
            dim=-1
        )

        # combine the values using attention
        output = attention_weights @ v

        return output


# combine multiple attention heads
class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_size,
        n_embd
    ):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size, n_embd)
            for _ in range(num_heads)
        ])


    def forward(self, x):

        outputs = [
            head(x)
            for head in self.heads
        ]

        # combine all the heads
        return torch.cat(outputs, dim=-1)


# regular neural network after attention
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                n_embd,
                4 * n_embd
            ),

            nn.ReLU(),

            nn.Linear(
                4 * n_embd,
                n_embd
            )
        )


    def forward(self, x):

        return self.net(x)


# one transformer block
class Block(nn.Module):

    def __init__(self, n_embd, num_heads):
        super().__init__()

        head_size = n_embd // num_heads

        self.attention = MultiHeadAttention(
            num_heads,
            head_size,
            n_embd
        )

        self.feed_forward = FeedForward(n_embd)


    def forward(self, x):

        # residual connection around attention
        x = x + self.attention(x)

        # residual connection around feed forward
        x = x + self.feed_forward(x)

        return x


# our tiny language model
class TinyLLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_embd,
        num_heads
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding = nn.Embedding(
            block_size,
            n_embd
        )

        self.transformer = Block(
            n_embd,
            num_heads
        )

        # turns embeddings into predictions for each character
        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )


    def forward(self, x):

        B, T = x.shape

        # get token embeddings
        token_emb = self.token_embedding(x)

        # get position embeddings
        pos_id = torch.arange(
            T,
            device=x.device
        )

        pos_emb = self.position_embedding(pos_id)

        # combine token and position information
        final_emb = token_emb + pos_emb

        # run through the transformer
        trans = self.transformer(final_emb)

        # predict the next character
        logits = self.lm_head(trans)

        return logits


# create the model
model = TinyLLM(
    vocab_size,
    n_embd,
    num_heads
)

print("\nNumber of parameters:")

print(
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# train the model
for i in range(num_steps):

    # pick random places in the text
    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,)
    )

    # create input sequences
    x_chunks = [
        data[start:start + block_size]
        for start in starts
    ]

    # same sequence but shifted by one character
    y_chunks = [
        data[start + 1:start + block_size + 1]
        for start in starts
    ]

    # turn them into batches
    x = torch.stack(x_chunks)
    y = torch.stack(y_chunks)

    # reset gradients
    optimizer.zero_grad()

    # get predictions
    logits = model(x)

    B, T = x.shape

    # flatten so cross entropy can calculate the loss
    logits = logits.reshape(
        B * T,
        vocab_size
    )

    targets = y.reshape(B * T)

    # see how wrong the predictions are
    loss = F.cross_entropy(
        logits,
        targets
    )

    # calculate gradients
    loss.backward()

    # update the model
    optimizer.step()

    if i % 50 == 0:
        print(
            f"Step {i}: Loss = {loss.item():.4f}"
        )


print("\nTraining finished!")
