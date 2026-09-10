import torch

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
    



