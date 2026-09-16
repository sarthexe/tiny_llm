import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):

    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()

        self.head_size = head_size

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        # prevents tokens from looking into the future
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # calculate attention scores
        attention_score = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)

        # hide future tokens
        masked_attention_score = attention_score.masked_fill(
            self.causal_mask[:T, :T] == 0,
            float("-inf")
        )

        # convert scores into probabilities
        attention_weights = F.softmax(masked_attention_score, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # combine the values using attention
        output = attention_weights @ v

        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outputs = [head(x) for head in self.heads]

        out = torch.cat(outputs, dim=-1)

        # mix information from all heads
        out = self.proj(out)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, num_heads, block_size, dropout=0.2):
        super().__init__()

        head_size = n_embd // num_heads

        self.attention = MultiHeadAttention(
            num_heads,
            head_size,
            n_embd,
            block_size,
            dropout
        )

        self.feed_forward = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # normalise first then attention then residual connection
        x = x + self.attention(self.ln1(x))

        # normalise first then feed forward then residual connection
        x = x + self.feed_forward(self.ln2(x))

        return x


class TinyLLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_embd=128,
        num_heads=4,
        num_layers=4,
        block_size=128,
        dropout=0.2
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)

        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.transformer = nn.Sequential(
            *[
                Block(n_embd, num_heads, block_size, dropout)
                for _ in range(num_layers)
            ]
        )

        # turns embeddings into predictions for each token
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        B, T = x.shape

        if T > self.block_size:
            raise ValueError(
                f"Input sequence length {T} is greater than block size {self.block_size}"
            )

        # get token embeddings
        token_emb = self.token_embedding(x)

        # get position embeddings
        pos_id = torch.arange(T, device=x.device)
        pos_emb = self.position_embedding(pos_id)

        # combine token and position information
        final_emb = token_emb + pos_emb

        # run through the transformer
        trans = self.transformer(final_emb)

        # predict the next token
        logits = self.lm_head(trans)

        return logits
