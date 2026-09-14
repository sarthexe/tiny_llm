import torch
import torch.nn as nn
import torch.nn.functional as F

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text[:500])
print("\nTotal characters:", len(text))

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

split_index = int(0.9 * len(data))

train_data = data[:split_index]
val_data = data[split_index:]

print("Total characters:", len(text))
print("Vocabulary size:", vocab_size)
print("Train size:", len(train_data))
print("Validation size:", len(val_data))
# model settings
block_size = 6
batch_size = 32

n_embd = 64
num_heads = 4

learning_rate = 0.001
num_steps = 5001


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

        self.dropout = nn.Dropout(0.2) 

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

        attention_weights = self.dropout(attention_weights)

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

        self.proj = nn.Linear(
            n_embd,
            n_embd
        )


    def forward(self, x):

        outputs = [
            head(x)
            for head in self.heads
        ]

        out = torch.cat(
            outputs,
            dim=-1
        )

        #mix information from all heads
        out = self.proj(out)
        
        return out


# regular neural network after attention
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                n_embd,
                4 * n_embd
            ),

            nn.GELU(),

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
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):

        # normalise first then attention then residual connection 
        x = x + self.attention(self.ln1(x))

        # normalise first then feed forward then residual connection 
        x = x + self.feed_forward(self.ln2(x))

        return x


# our tiny language model
class TinyLLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_embd,
        num_heads,
        num_layers=4
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

        self.transformer = nn.Sequential(
            *[
                Block(n_embd,num_heads)
                for _ in range(num_layers)
            ]
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

print("\nParameters by layer:\n")

for name, param in model.named_parameters():
    print(
        f"{name:60} {param.numel():>8}"
    )


# optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

def get_batch(data_source):
    starts = torch.randint(
        0,
        len(data_source) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data_source[start:start + block_size]
        for start in starts
    ])

    y = torch.stack([
        data_source[start + 1:start + block_size + 1]
        for start in starts
    ])

    return x, y


def estimate_loss():
    losses = {}

    # evaluate both datasets
    for split,data_source in  [
        ("train", train_data),
        ("val",val_data)
    ]:
        model.eval()
        split_losses=[]
        with torch.no_grad():
            for _ in range(20):
                x,y = get_batch(data_source)

                logits = model(x)

                B,T = x.shape

                logits = logits.reshape(
                    B*T,
                    vocab_size
                )

                targets = y.reshape(B*T)

                loss = F.cross_entropy(
                    logits,
                    targets
                )

                split_losses.append(loss.item())

        losses[split] = sum(split_losses)/len(split_losses)

    model.train()

    return losses



#dropout enabled
model.train()

# train the model
for i in range(num_steps):

    x, y = get_batch(train_data)

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

    if i % 500 == 0:

        losses = estimate_loss()

        print(
            f"Step {i} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Val Loss: {losses['val']:.4f}"
        )


print("\nTraining finished!")

context = torch.tensor(
    [[stoi["t"]]],
    dtype=torch.long
)

#dropout disabled
model.eval()
with torch.no_grad():

    temperature = 0.8
    #autoregressive generation
    for _ in range(200):

        # only give model the last 8 tokens
        context_for_model = context[:,-block_size:] #  : means take all batches -block_size means start block_size positions from the end and take everything until the end.

        #get predictions
        logits = model(context_for_model)
        # only use the prediction from the last position
        logits = logits[:,-1,:]

        #control randomness
        logits = logits/temperature

        #convert the logits into probabilities
        probs = F.softmax(logits,dim=-1)

        # randomly sample the next token
        next_token = torch.multinomial(probs, num_samples=1) #multinomial means to pick a token according to these probs

        # add it to your full context
        context = torch.cat(
            (context,next_token),
            dim=1
        )


    generated_text = ''.join(
        itos[token.item()]
        for token in context[0]
    )

    print(generated_text)


