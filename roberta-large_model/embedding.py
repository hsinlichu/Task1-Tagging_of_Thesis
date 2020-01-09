import torch
from tqdm import tqdm
import logging


class fasttext_embedding:
    def __init__(self, rawdata_path, seed=1357):
        self.word_dict = {}
        self.vectors = []
        torch.manual_seed(seed)
        self.load_embedding(rawdata_path)

    
    def load_embedding(self, rawdata_path):
        print("load_embedding")
        with open(rawdata_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:          # skip header
                    continue

                row = line.rstrip().split(' ') # rstrip() method removes any trailing characters (default space)
                word, vector = row[0], row[1:]
                word = word.lower()
                if word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)
                    vector = [float(n) for n in vector] 
                    self.vectors.append(vector)
                #print(self.word_dict)
                #print(self.vectors)
                #print(self.get_dim())
        self.vectors = torch.tensor(self.vectors)

        if '<unk>' not in self.word_dict:
            self.add('<unk>')
        if '<pad>' not in self.word_dict:
            self.add('<pad>', torch.zeros(1, self.get_dim()))
        print("done")

        #logging.info("Embedding size: {}".format(self.vectors.size()))

    def get_dim(self):
        return len(self.vectors[0])

    def add(self, word, vector=None):
        if vector is None:
            vector = torch.empty(1,self.get_dim())
            torch.nn.init.uniform_(vector)
        vector.view(1,-1)
        self.word_dict[word] = len(self.word_dict)
        self.vectors = torch.cat((self.vectors, vector), 0)

    def to_index(self, word):
        word = word.lower()
        if word in self.word_dict:
            return self.word_dict[word] 
        else:
            return self.word_dict['<unk>']

