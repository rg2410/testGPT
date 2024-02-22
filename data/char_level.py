class CharEncDec:

    def __init__(self, text):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for i,ch in enumerate(self.chars)}

    def encode(self, to_encode):
        return [self.stoi[c] for c in to_encode]
    
    def decode(self, to_decode):
        return ''.join([self.itos[c] for c in to_decode])
