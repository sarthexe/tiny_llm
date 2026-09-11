import torch
import torch.nn as nn
import torch.nn.functional as F

text = """
the cat sat on the mat.
the dog sat on the floor.
the cat likes the dog.
"""

# 1. Create our vocabulary
chars = sorted(list(set(text)))

print("Vocabulary:", chars)
print("Vocabulary size:", len(chars))

# 2. Map characters -> integers
stoi = {ch: i for i, ch in enumerate(chars)}

# 3. Map integers -> characters
itos = {i: ch for i, ch in enumerate(chars)}

# 4. Tokenize the text
encoded = [stoi[ch] for ch in text]

print("Original:")
print(text)

print("Encoded:")
print(encoded)

# 5. Convert to a PyTorch tensor
data = torch.tensor(encoded, dtype=torch.long)

print("Tensor:")
print(data)


block_size = 8

x = data[:block_size]

y = data[1:block_size+1]

print("Input:", x)
print("Target:", y)


for i,_ in enumerate(x):
    ans = x[:i+1]
    dec_ans = ''.join(itos[token.item()] for token in ans)

    tar = y[i]
    dec_tar = itos[tar.item()]

    print("Context:", repr(dec_ans))
    print("Target:", repr(dec_tar))
    print()
    
#embedding layer
vocab_size = len(chars)

n_embd = 16

embedding = nn.Embedding(vocab_size,n_embd)

token_embedding = embedding(x)

print(x)
print(token_embedding)
print(token_embedding.shape)

#positional embeddings
pos_id = torch.arange(block_size)

pos_emb = nn.Embedding(block_size,n_embd)

position_embeddings = pos_emb(pos_id)

#final embedding  =  tokens embedding + positional Embeddings
final_emb = token_embedding + position_embeddings

print("Token Embedding Shape:" ,token_embedding.shape)
print("Positional Embedding Shape:",position_embeddings.shape)
print("Final Embedding Shape:",final_emb.shape)

#self attention
head_size = 16
query = nn.Linear(n_embd, head_size, bias=False)
key = nn.Linear(n_embd, head_size, bias=False)
value = nn.Linear(n_embd, head_size, bias=False)


q = query(final_emb)
k = key(final_emb)
v = value(final_emb)

print("Q:",q.shape)
print("K:",k.shape)
print("V:",v.shape)


#attention score = Q * K Transpose
attention_score = (q@k.T)/head_size**0.5 

print("Attention Score: \n",attention_score)
print("Attention Score's Shape: ",attention_score.shape)

#causal mask
causal_mask = torch.tril(torch.ones(block_size,block_size))

print("causal mask: ",causal_mask)
print("causal mask's shape: ",causal_mask.shape)

#masked attention score 
masked_attention_score = attention_score.masked_fill(
    causal_mask == 0,
    float('-inf')
)
print("masked attention score: \n", masked_attention_score)

#Applying softmax to the masked attention score

attention_weights = F.softmax(masked_attention_score, dim=-1)
print("Attention weights: \n",attention_weights)


#Single attention head's output 
attention_output = attention_weights @ v

print("Attention Output: \n", attention_output)
print("Attention Output's shape: \n", attention_output.shape)


class Head(nn.Module):
    def __init__(self, head_size,n_embd):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size,bias=False)
        self.query = nn.Linear(n_embd, head_size,bias=False)
        self.value = nn.Linear(n_embd, head_size,bias=False)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size,block_size))
        )

    def forward(self,x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_score = (q@k.T) / (self.head_size**0.5)
        masked_attention_score = attention_score.masked_fill(
            self.causal_mask == 0,
            float('-inf')
        )
        attention_weights = F.softmax(masked_attention_score,dim=-1)
        output = attention_weights @ v

        return output



head = Head(head_size,n_embd)

res = head(final_emb)

print("res: ",res)
print("res's shape: ",res.shape)




    