import torch
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pickle
import nltk
nltk.download('punkt')


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



class ThesisTaggingDataLoader(BaseDataLoader):
    """
    ThesisTagging data loading demo using BaseDataLoader
    """
    def __init__(self, train_data_path, test_data_path, batch_size, num_classes, embedding, padding="<pad>", padded_len=40,
            shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if training:
            data_path = train_data_path
        else:
            data_path = test_data_path
            
        self.dataset = ThesisTaggingDataset(data_path, embedding, num_classes, padding, padded_len, training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.dataset.collate_fn)

    


class ThesisTaggingDataset(Dataset):
    def __init__(self, data_path, embedding, num_classes, padding, padded_len, training):
        self.embedding = embedding
        self.data_path = data_path
        self.padded_len = padded_len
        self.num_classes = num_classes
        self.padding = self.embedding.to_index(padding)
        self.training = training

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.dataset = []
        for i in data:
            self.dataset += i
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index] 
        sentence_indice = self.sentence_to_indices(data["sentence"])
        if self.training:
            return [data["number"], sentence_indice, data["label"]]
        else:
            return [data["number"], sentence_indice]

    def tokenize(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        #print(tokens)

        return tokens


    def sentence_to_indices(self, sentence):
        sentence = sentence.lower()
        sentence = self.tokenize(sentence)

        ret = []
        for word in sentence:
            ret.append(self.embedding.to_index(word))
        return ret

    def _pad_to_len(self, arr):
        if len(arr) > self.padded_len:
            arr = arr[:self.padded_len]
        while len(arr) < self.padded_len:
            arr.append(self.padding)
        return arr

    def _to_one_hot(self, y):
        ret = [0 for i in range(self.num_classes)]
        for l in y:
            ret[l] = 1
        return ret
    def collate_fn(self, datas):
        batch = {}
        if self.training:
            batch['label'] = torch.tensor([ self._to_one_hot(r[2]) for r in datas])
        batch['sentence'] = torch.tensor([ self._pad_to_len(r[1]) for r in datas])
        batch['number'] = [ r[0] for r in datas]

        return batch
class ThesisTaggingArticleDataset(Dataset):
    def __init__(self, data_path, embedding, num_classes, padding, padded_len, training):
        self.embedding = embedding
        self.data_path = data_path
        self.padded_len = padded_len
        self.num_classes = num_classes
        self.padding = self.embedding.to_index(padding)
        self.training = training

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.dataset = []
        for i in data:
            self.dataset += i
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index] 
        sentence_indice = self.sentence_to_indices(data["sentence"])
        if self.training:
            return [data["number"], sentence_indice, data["label"]]
        else:
            return [data["number"], sentence_indice]



