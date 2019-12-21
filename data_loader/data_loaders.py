import torch
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pickle
import nltk
nltk.download('punkt')

import pprint
pp = pprint.PrettyPrinter(indent=4)

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


class ThesisTaggingDataLoader_bert(BaseDataLoader):
    """
    ThesisTagging data loading demo using BaseDataLoader
    """
    def __init__(self, train_data_path, test_data_path, batch_size, num_classes, embedding, padding="<pad>", padded_len=40,
            shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if training:
            data_path = train_data_path
        else:
            data_path = test_data_path
            
        self.dataset = ThesisTaggingDataset_bert(data_path, embedding, num_classes, padding, padded_len, training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.dataset.collate_fn)

class ThesisTaggingDataset_bert(Dataset):
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
        sentence_indice = data["sentence"]
        if self.training:
            return [data["number"], sentence_indice, data["label"]]
        else:
            return [data["number"], sentence_indice]

    def _to_one_hot(self, y):
        ret = [0 for i in range(self.num_classes)]
        for l in y:
            ret[l] = 1
        return ret

    def collate_fn(self, datas):
        batch = {}
        batch['label'] = []
        batch['sentence'] = []
        batch['number'] = []

        for r in datas:
            if self.training:
                batch['label'].append(self._to_one_hot(r[2]))
            batch['sentence'].append(r[1]) 
            batch['number'].append(r[0])
        batch['label'] = torch.tensor(batch['label'])


        return batch



class ThesisTaggingArticleDataLoader(BaseDataLoader):
    """
    ThesisTagging data loading demo using BaseDataLoader
    """
    def __init__(self, train_data_path, test_data_path, batch_size, num_classes, embedding, padding="<pad>", padded_len=40,
            shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if training:
            data_path = train_data_path
        else:
            data_path = test_data_path
            
        self.dataset = ThesisTaggingArticleDataset(data_path, embedding, num_classes, padding, padded_len, training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.dataset.collate_fn)



class ThesisTaggingArticleDataset(ThesisTaggingDataset):
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
            self.dataset.append(i)
        print("Number of training data: {}".format(len(self.dataset)))

    def __getitem__(self, index):
        data = self.dataset[index] 
        for i in range(len(data)):
            data[i]["sentence"] = self.sentence_to_indices(data[i]["sentence"])
        return data

    def collate_fn(self, datas):
        batch = {}
        batch['label'] = []
        batch['sentence'] = []
        batch['number'] = []

        for article in datas:
            article_number = []
            article_label = []
            article_sentence = []
            if self.training:
                article_label = [ self._to_one_hot(r["label"]) for r in article]
            article_sentence = [ self._pad_to_len(r["sentence"]) for r in article]
            article_number = [ r["number"] for r in article]
            batch['label'].append(torch.LongTensor(article_label))
            batch['sentence'].append(torch.LongTensor(article_sentence))
            batch['number'].append(article_number)

        return batch

class ThesisTaggingArticleDataLoader_bert(BaseDataLoader):
    """
    ThesisTagging data loading demo using BaseDataLoader
    """
    def __init__(self, train_data_path, test_data_path, batch_size, num_classes, embedding, padding="<pad>", padded_len=40,
            shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if training:
            data_path = train_data_path
        else:
            data_path = test_data_path
            
        self.dataset = ThesisTaggingArticleDataset_bert(data_path, embedding, num_classes, padding, padded_len, training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.dataset.collate_fn)



class ThesisTaggingArticleDataset_bert(ThesisTaggingDataset):
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
            self.dataset.append(i)
        print("Number of training data: {}".format(len(self.dataset)))

    def __getitem__(self, index):
        data = self.dataset[index] 
        return data

    def collate_fn(self, datas):
        batch = {}
        batch['label'] = []
        batch['sentence'] = []
        batch['number'] = []

        for article in datas:
            article_number = []
            article_label = []
            article_sentence = []
            if self.training:
                article_label = [ self._to_one_hot(r["label"]) for r in article]
            article_sentence = [ r["sentence"] for r in article]
            article_number = [ r["number"] for r in article]
            batch['label'].append(torch.LongTensor(article_label))
            batch['sentence'].append(article_sentence)
            batch['number'].append(article_number)

        return batch
