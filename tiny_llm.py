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

