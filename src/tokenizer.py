from collections import Counter


class ByteBPETokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.merges = {}

    def train(self, text, num_merges):
        # start with all possible bytes
        self.stoi = {bytes([i]): i for i in range(256)}
        self.itos = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        tokens = list(text.encode("utf-8"))

        for merge_num in range(num_merges):
            pairs = Counter(zip(tokens, tokens[1:]))
            if not pairs:
                break

            pair, count = pairs.most_common(1)[0]
            new_token_id = len(self.itos)
            new_token = self.itos[pair[0]] + self.itos[pair[1]]

            self.stoi[new_token] = new_token_id
            self.itos[new_token_id] = new_token
            self.merges[pair] = {"id": new_token_id, "rank": merge_num}

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

            print(
                f"Merge {merge_num + 1}: "
                f"{self.itos[pair[0]]} + {self.itos[pair[1]]} "
                f"-> {new_token} ({count} times)"
            )

    def encode(self, text):
        tokens = list(text.encode("utf-8"))

        while True:
            available = [
                pair for pair in zip(tokens, tokens[1:]) if pair in self.merges
            ]
            if not available:
                break

            pair = min(available, key=lambda p: self.merges[p]["rank"])
            new_token_id = self.merges[pair]["id"]

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def decode(self, tokens):
        byte_data = b"".join(self.itos[token] for token in tokens)
        return byte_data.decode("utf-8")
