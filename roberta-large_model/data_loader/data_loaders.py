import torch
from transformers import *
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



class ThesisTaggingDataLoaderbert(BaseDataLoader):
    """
    ThesisTagging data loading demo using BaseDataLoader
    """
    def __init__(self, train_data_path, test_data_path, batch_size, num_classes, embedding, padding="<pad>", padded_len=40,
            shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if training:
            data_path = train_data_path
        else:
            data_path = test_data_path
            
        self.dataset = ThesisTaggingDatasetbert(data_path, embedding, num_classes, padding, padded_len, training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.dataset.collate_fn)

    


class ThesisTaggingDatasetbert(Dataset):
    def __init__(self, data_path, embedding, num_classes, padding, padded_len, training):
        #self.embedding = embedding
        self.data_path = data_path
        self.padded_len = padded_len
        self.num_classes = num_classes
        #self.padding = self.embedding.to_index(padding)
        self.training = training

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.dataset = []
        for i in data:
            self.dataset += i


        #edit by wilber
        MODELS = [
          (BertModel,       BertTokenizer,       'bert-base-uncased'),
          ]

        # To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

        # Let's encode some text in a sequence of hidden-states using each model:
        for model_class, tokenizer_class, pretrained_weights in MODELS:
            # Load pretrained model/tokenizer
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            self.model = model_class.from_pretrained(pretrained_weights)
        #edit by wilber
        self.padding = self.tokenizer.encode(padding)[0]


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index] 
        sentence_indice = self.sentence_to_indices(data["sentence"])
        if self.training:
            return [data["number"], sentence_indice, data["label"]]
        else:
            return [data["number"], sentence_indice]
    #no use
    def tokenize(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        #print(tokens)

        return tokens


    def sentence_to_indices(self, sentence):
        sentence = sentence.lower()
        #sentence = self.tokenize(sentence)
        ret = self.tokenizer.encode(sentence, add_special_tokens=True)
        '''
        ret = []
        for word in sentence:
            ret.append(self.embedding.to_index(word))
        '''
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




####################XLNET#####################


class ThesisTaggingDataLoaderxlnet(BaseDataLoader):
    """
    ThesisTagging data loading demo using BaseDataLoader
    """
    def __init__(self, train_data_path, test_data_path, batch_size, num_classes, embedding, padding="<pad>", padded_len=40,
            shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if training:
            data_path = train_data_path
        else:
            data_path = test_data_path
            
        self.dataset = ThesisTaggingDatasetxlnet(data_path, embedding, num_classes, padding, padded_len, training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.dataset.collate_fn)

    


class ThesisTaggingDatasetxlnet(Dataset):
    def __init__(self, data_path, embedding, num_classes, padding, padded_len, training):
        #self.embedding = embedding
        self.data_path = data_path
        self.padded_len = padded_len
        self.num_classes = num_classes
        #self.padding = self.embedding.to_index(padding)
        self.training = training

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.dataset = []
        for i in data:
            self.dataset += i


        #edit by wilber
        MODELS = [
          (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          ]

        # To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

        # Let's encode some text in a sequence of hidden-states using each model:
        for model_class, tokenizer_class, pretrained_weights in MODELS:
            # Load pretrained model/tokenizer
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            self.model = model_class.from_pretrained(pretrained_weights)
        #edit by wilber
        self.padding = self.tokenizer.encode(padding)[0]


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
        #sentence = sentence.lower()
        #sentence = self.tokenize(sentence)
        ret = self.tokenizer.encode(sentence, add_special_tokens=True)
        '''
        ret = []
        for word in sentence:
            ret.append(self.embedding.to_index(word))
        '''
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

####################roberta#####################


class ThesisTaggingDataLoaderroberta(BaseDataLoader):
    """
    ThesisTagging data loading demo using BaseDataLoader
    """
    def __init__(self, train_data_path, test_data_path, batch_size, num_classes, embedding, padding="<pad>", padded_len=40,
            shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if training:
            data_path = train_data_path
        else:
            data_path = test_data_path
            
        self.dataset = ThesisTaggingDatasetroberta(data_path, embedding, num_classes, padding, padded_len, training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.dataset.collate_fn)

    


class ThesisTaggingDatasetroberta(Dataset):
    def __init__(self, data_path, embedding, num_classes, padding, padded_len, training):
        #self.embedding = embedding
        self.data_path = data_path
        self.padded_len = padded_len
        self.num_classes = num_classes
        #self.padding = self.embedding.to_index(padding)
        self.training = training

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.dataset = []
        for i in data:
            self.dataset += i


        #edit by wilber
        MODELS = [
          (RobertaModel,    RobertaTokenizer,    'roberta-base'),
          ]

        # To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

        # Let's encode some text in a sequence of hidden-states using each model:
        for model_class, tokenizer_class, pretrained_weights in MODELS:
            # Load pretrained model/tokenizer
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            #self.model = model_class.from_pretrained(pretrained_weights)
        #edit by wilber
        self.padding = self.tokenizer.encode(padding)[0]
        print("padding")
        print(self.padding)
        print(self.tokenizer.encode('[PAD]')[0]) 
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
        #sentence = sentence.lower()
        #sentence = self.tokenize(sentence)
        ret = self.tokenizer.encode(sentence, add_special_tokens=True)
        '''
        ret = []
        for word in sentence:
            ret.append(self.embedding.to_index(word))
        '''
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




####################XLM#####################


class ThesisTaggingDataLoaderxlm(BaseDataLoader):
    """
    ThesisTagging data loading demo using BaseDataLoader
    """
    def __init__(self, train_data_path, test_data_path, batch_size, num_classes, embedding, padding="<pad>", padded_len=40,
            shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if training:
            data_path = train_data_path
        else:
            data_path = test_data_path
            
        self.dataset = ThesisTaggingDatasetxlm(data_path, embedding, num_classes, padding, padded_len, training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.dataset.collate_fn)

    


class ThesisTaggingDatasetxlm(Dataset):
    def __init__(self, data_path, embedding, num_classes, padding, padded_len, training):
        #self.embedding = embedding
        self.data_path = data_path
        self.padded_len = padded_len
        self.num_classes = num_classes
        #self.padding = self.embedding.to_index(padding)
        self.training = training

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.dataset = []
        for i in data:
            self.dataset += i


        #edit by wilber
        MODELS = [
          (XLMModel,        XLMTokenizer,        'xlm-mlm-en-2048'),
          ]

        # To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

        # Let's encode some text in a sequence of hidden-states using each model:
        for model_class, tokenizer_class, pretrained_weights in MODELS:
            # Load pretrained model/tokenizer
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            self.model = model_class.from_pretrained(pretrained_weights)
        #edit by wilber
        self.padding = self.tokenizer.encode(padding)[0]


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
        #sentence = sentence.lower()
        #sentence = self.tokenize(sentence)
        ret = self.tokenizer.encode(sentence, add_special_tokens=True)
        '''
        ret = []
        for word in sentence:
            ret.append(self.embedding.to_index(word))
        '''
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

####################Concade#####################





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
        #self.embedding = embedding
        self.data_path = data_path
        self.padded_len = padded_len
        self.num_classes = num_classes
        #self.padding = self.embedding.to_index(padding)
        self.training = training

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.dataset = []
        for i in data:
            self.dataset.append(i)
        

        #edit by wilber
        MODELS = [
          (RobertaModel,    RobertaTokenizer,    'roberta-large'),
          ]

        # To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

        # Let's encode some text in a sequence of hidden-states using each model:
        for model_class, tokenizer_class, pretrained_weights in MODELS:
            # Load pretrained model/tokenizer
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            self.model = model_class.from_pretrained(pretrained_weights)
        #edit by wilber
        self.padding = self.tokenizer.encode(padding)[0]



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
        #sentence = sentence.lower()
        #sentence = self.tokenize(sentence)
        ret = self.tokenizer.encode(sentence, add_special_tokens=True)
        '''ret = []
        for word in sentence:
            ret.append(self.embedding.to_index(word))'''
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
        #self.embedding = embedding
        self.data_path = data_path
        self.padded_len = padded_len
        self.num_classes = num_classes
        #self.padding = self.embedding.to_index(padding)
        self.training = training

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.dataset = []
        for i in data:
            self.dataset.append(i)

        MODELS = [
          (RobertaModel,    RobertaTokenizer,    'roberta-large'),##roberta large
          ]

        # To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

        # Let's encode some text in a sequence of hidden-states using each model:
        for model_class, tokenizer_class, pretrained_weights in MODELS:
            # Load pretrained model/tokenizer
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            self.model = model_class.from_pretrained(pretrained_weights)
        #edit by wilber
        self.padding = self.tokenizer.encode(padding)[0]
        print("Number of training data: {}".format(len(self.dataset)))
    
    def sentence_to_indices(self, sentence):
        #sentence = sentence.lower()
        #sentence = self.tokenize(sentence)
        ret = self.tokenizer.encode(sentence, add_special_tokens=True)
        '''ret = []
        for word in sentence:
            ret.append(self.embedding.to_index(word))'''
        return ret

    def __getitem__(self, index):
        data = self.dataset[index] 
        for i in range(len(data)):
            data[i]["sentence"] = self.sentence_to_indices(data[i]["sentence"])

        return data

    def _pad_to_len(self, arr):
        if len(arr) > self.padded_len:
            arr = arr[:self.padded_len]
            #arr = self.sentence_to_indices(arr)
        #elif len(arr) == self.padded_len:
        #    arr = self.sentence_to_indices(arr)
        #else:
        #    arr = self.sentence_to_indices(arr)
        while len(arr) < self.padded_len:
            arr.append(self.padding)
        return arr
        
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
