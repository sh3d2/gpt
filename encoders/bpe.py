import pickle


class BpeTokenizer:
    def __init__(self, vocab_size, dict={}):
        self.dict = dict
        self.__vocab_size = vocab_size

    @classmethod
    def from_file(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            tokenizer = BpeTokenizer(data['vocab_size'], data['dict'])
            return tokenizer

    def train(self, text):
        tokens = list(map(int, text.encode('utf-8')))
        num_merges = self.__vocab_size - 256
        ids = list(tokens)

        for i in range(num_merges):
            stats = self.__get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.__merge(ids, pair, idx)
            self.dict[pair] = idx


    def to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'vocab_size': self.__vocab_size,
                'dict': self.dict
            }, f)

    def __get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def __merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def encode(self, text):
        pass

    def decode(self, tokens):
        pass

    def vocab_size(self):
        return self.__vocab_size
