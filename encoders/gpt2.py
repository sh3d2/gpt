import tiktoken


class Gpt2Tokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding('gpt2')

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def vocab_size(self):
        return self.tokenizer.n_vocab
