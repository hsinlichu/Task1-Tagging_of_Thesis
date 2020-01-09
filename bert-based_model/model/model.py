import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from transformers import *

import os

class ThesisTaggingModel(BaseModel):
    def __init__(self, dim_embeddings, num_classes, embedding, hidden_size=128,
            num_layers=1,  rnn_dropout=0.2, clf_dropout=0.3, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_embeddings = dim_embeddings
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bool(bidirectional)
        self.clf_dropout = clf_dropout

        #logging.info("Embedding size: ({},{})".format(embedding.size(0),embedding.size(1)))
        self.embedding = nn.Embedding(embedding.vectors.size(0), embedding.vectors.size(1))
        self.embedding.weight = nn.Parameter(embedding.vectors)

        self.rnn = nn.LSTM(input_size=self.dim_embeddings, hidden_size=self.hidden_size,
                num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True) # , dropout=rnn_dropout
        self.clf = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.clf_dropout),
                nn.Linear(hidden_size // 2, num_classes),
                nn.Sigmoid()
                )

    def forward(self, sentence):
        with torch.no_grad():
            sentence = self.embedding(sentence)
        #print(sentence.size()) # torch.Size([128, 30, 300])
        sentence_out, hidden = self.rnn(sentence)

        last_output = sentence_out[:,-1,:]

        score = self.clf(last_output)
        return score

class ThesisClassificationModel_hierarchical(BaseModel):
    def __init__(self, dim_embeddings, num_classes, embedding, hidden_size=128,
            num_layers=1,  rnn_dropout=0.2, clf_dropout=0.3, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_embeddings = dim_embeddings
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bool(bidirectional)
        self.clf_dropout = clf_dropout

        #logging.info("Embedding size: ({},{})".format(embedding.size(0),embedding.size(1)))
        self.embedding = nn.Embedding(embedding.vectors.size(0), embedding.vectors.size(1))
        self.embedding.weight = nn.Parameter(embedding.vectors)

        self.rnn = nn.LSTM(input_size=self.dim_embeddings, hidden_size=self.hidden_size,
                num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True) # , dropout=rnn_dropout
        
        self.clf2_linear1 = nn.Linear(self.hidden_size, hidden_size // 2)
        self.clf2_linear12 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear2 = nn.Linear(self.hidden_size+ 1, hidden_size // 2)
        self.clf2_linear22 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear3 = nn.Linear(self.hidden_size+ 2, hidden_size // 2)
        self.clf2_linear32 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear4 = nn.Linear(self.hidden_size+ 3, hidden_size // 2)
        self.clf2_linear42 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear5 = nn.Linear(self.hidden_size+ 4, hidden_size // 2)
        self.clf2_linear52 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear6 = nn.Linear(self.hidden_size+ 5, hidden_size // 2)
        self.clf2_linear62 = nn.Linear(hidden_size // 2, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, sentence):
        with torch.no_grad():
            sentence = self.embedding(sentence)
        #print(sentence.size()) # torch.Size([128, 30, 300])
        sentence_out, hidden = self.rnn(sentence)
        last_output = sentence_out[:,-1,:]
        #print(last_output.size())


        x = self.clf2_linear1(last_output)
        x = self.clf2_linear12(x)
        x = self.Sigmoid(x)
        x2 = torch.cat((last_output,x),1)
        y = self.clf2_linear2(x2)
        y = self.clf2_linear22(y)
        y = self.Sigmoid(y)
        y2 = torch.cat((x2,y),1)
        i = self.clf2_linear3(y2)
        i = self.clf2_linear32(i)
        i = self.Sigmoid(i)
        i2 = torch.cat((y2,i),1)
        j = self.clf2_linear4(i2)
        j = self.clf2_linear42(j)
        j = self.Sigmoid(j)
        j2 = torch.cat((i2,j),1)
        k = self.clf2_linear5(j2)
        k = self.clf2_linear52(k)
        k = self.Sigmoid(k)
        k2 = torch.cat((j2,k),1)
        m = self.clf2_linear6(k2)
        m = self.clf2_linear62(m)
        m = self.Sigmoid(m)
        score = torch.cat((x,y,i,j,k,m),1)        

        return score

class ThesisTaggingModel_cascade(BaseModel):
    def __init__(self, dim_embeddings, num_classes, embedding, hidden_size=128,
            num_layers=1,  rnn_dropout=0.2, clf_dropout=0.3, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_embeddings = dim_embeddings
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bool(bidirectional)
        self.clf_dropout = clf_dropout

        #logging.info("Embedding size: ({},{})".format(embedding.size(0),embedding.size(1)))
        self.embedding = nn.Embedding(embedding.vectors.size(0), embedding.vectors.size(1))
        self.embedding.weight = nn.Parameter(embedding.vectors)

        #self.rnn = nn.LSTM(input_size=self.dim_embeddings, hidden_size=self.hidden_size,
        self.rnn = nn.GRU(input_size=self.dim_embeddings, hidden_size=self.hidden_size,
                num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True) # , dropout=rnn_dropout

        self.clf2_linear1 = nn.Linear(self.hidden_size + self.num_classes, hidden_size // 2)
        #self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.clf2_linear12 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear2 = nn.Linear(self.hidden_size + self.num_classes+ 1, hidden_size // 2)
        #self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.clf2_linear22 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear3 = nn.Linear(self.hidden_size + self.num_classes+ 2, hidden_size // 2)
        #self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.clf2_linear32 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear4 = nn.Linear(self.hidden_size + self.num_classes+ 3, hidden_size // 2)
        #self.bn4 = nn.BatchNorm1d(hidden_size // 2)
        self.clf2_linear42 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear5 = nn.Linear(self.hidden_size + self.num_classes+ 4, hidden_size // 2)
        #self.bn5 = nn.BatchNorm1d(hidden_size // 2)
        self.clf2_linear52 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear6 = nn.Linear(self.hidden_size + self.num_classes+ 5, hidden_size // 2)
        #self.bn6 = nn.BatchNorm1d(hidden_size // 2)
        self.clf2_linear62 = nn.Linear(hidden_size // 2, 1)
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        #self.dpo = nn.Dropout(p=self.clf_dropout)

    def submodel(self, former_output,  sentence):
        former_output = former_output.to("cuda")
        sentence = sentence.to("cuda")
        with torch.no_grad():
            sentence = self.embedding(sentence)
        #print(sentence.size()) # torch.Size([128, 30, 300])
        sentence_out, hidden = self.rnn(sentence)
        last_output = sentence_out[:,-1,:]

        x1 = torch.cat((former_output, last_output),1)
        x = self.clf2_linear1(x1)
        #x = self.bn1(x)
        x = self.relu(x)
        #x = self.dpo(x)
        x = self.clf2_linear12(x)
        x = self.Sigmoid(x)
        x2 = torch.cat((x1,x),1)
        y = self.clf2_linear2(x2)
        #y = self.bn2(y)
        y = self.relu(y)
        #y = self.dpo(y)
        y = self.clf2_linear22(y)
        y = self.Sigmoid(y)
        y2 = torch.cat((x2,y),1)
        i = self.clf2_linear3(y2)
        #i = self.bn3(i)
        i = self.relu(i)
        #i = self.dpo(i)
        i = self.clf2_linear32(i)
        i = self.Sigmoid(i)
        i2 = torch.cat((y2,i),1)
        j = self.clf2_linear4(i2)
        #j = self.bn4(j)
        j = self.relu(j)
        #j = self.dpo(j)
        j = self.clf2_linear42(j)
        j = self.Sigmoid(j)
        j2 = torch.cat((i2,j),1)
        k = self.clf2_linear5(j2)
        #k = self.bn5(k)
        k = self.relu(k)
        #k = self.dpo(k)
        k = self.clf2_linear52(k)
        k = self.Sigmoid(k)
        k2 = torch.cat((j2,k),1)
        m = self.clf2_linear6(k2)
        #m = self.bn6(m)
        m = self.relu(m)
        #m = self.dpo(m)
        m = self.clf2_linear62(m)
        m = self.Sigmoid(m)
        score = torch.cat((x,y,i,j,k,m),1)        

        return score

    def forward(self, batch):

        sentence_per_article = torch.IntTensor([article.size(0) for article in batch])
        max_sentence_per_article = torch.max(sentence_per_article).item()
        #print(sentence_per_article)
        #print(max_sentence_per_article)


        for i in range(len(batch)):
            batch[i] = F.pad(batch[i], (0, 0, 0, max_sentence_per_article - batch[i].size(0)), "constant", 0)
        batch = torch.stack(batch).to("cuda")

        output = []
        init = torch.zeros([ batch.size(0), self.num_classes])
        former_output = init
        for i in range(max_sentence_per_article):
            score = self.submodel( former_output, batch[:, i, :])
            output.append(score)
            former_output = score

        output = torch.stack(output, dim=1)
        ret = []
        for i in range(output.size(0)):
            ret.append(output[i][:sentence_per_article[i]])
        return ret
    

class ThesisTaggingModel_bert(BaseModel):
    def __init__(self, dim_embeddings, num_classes, embedding, hidden_size=128,
            num_layers=1,  rnn_dropout=0.2, clf_dropout=0.3, bidirectional=False):
        super().__init__()
        self.hidden_size = 768
        self.dim_embeddings = dim_embeddings
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bool(bidirectional)
        self.clf_dropout = clf_dropout

        MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
          (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          (CTRLModel,       CTRLTokenizer,       'ctrl'),
          (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          (RobertaModel,    RobertaTokenizer,    'roberta-base')]
        
        
        model_class, tokenizer_class, pretrained_weights = MODELS[8]
        weight = pretrained_weights
        weight_dir = pretrained_weights

        #weight_dir = "../Task2-Classification_of_Thesis/lmft/"
        #weight = os.path.join(weight_dir, "checkpoint-23000")
        print(weight)
        self.tokenizer = tokenizer_class.from_pretrained(weight_dir)
        self.bert_model = model_class.from_pretrained(weight)

        self.padded_len = 60
        self.pad_id = self.tokenizer.encode("[PAD]")[0]
        print("pad_id", self.pad_id)

        self.clf = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.BatchNorm1d(self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.clf_dropout),
                nn.Linear(self.hidden_size // 2, num_classes),
                nn.Sigmoid()
                )

    def _pad_to_len(self, arr):
        if len(arr) > self.padded_len:
            arr = arr[:self.padded_len]
        while len(arr) < self.padded_len:
            arr.append(self.pad_id)
            
        return arr


    def forward(self, sentence):
        #with torch.no_grad():
        input_id = torch.tensor([self._pad_to_len(self.tokenizer.encode(sent, add_special_tokens=True)) # add [CLS] [PAD]
                for sent in sentence]).to("cuda")
        attention_mask = (input_id != self.pad_id).float()
        #print(input_id.size()) # [128, 40]
        first_output = self.bert_model(input_id, attention_mask=attention_mask)[0][:,0,:]
        #print(first_output.size()) # [128, 768]

        score = self.clf(first_output)
        return score

class ThesisTaggingModel_cascade_bert_3sent(BaseModel):
    def __init__(self, dim_embeddings, num_classes, embedding, hidden_size=768,
            num_layers=1,  rnn_dropout=0.2, clf_dropout=0.3, bidirectional=False):
        super().__init__()
        self.hidden_size = 768
        self.dim_embeddings = dim_embeddings
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bool(bidirectional)
        self.clf_dropout = clf_dropout
        self.fp16 = False
        

        MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
          (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          (CTRLModel,       CTRLTokenizer,       'ctrl'),
          (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          (RobertaModel,    RobertaTokenizer,    'roberta-base'),
          (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base' ),
          (T5Model, T5Tokenizer, 't5-base')


          ]
        
        
        model_class, tokenizer_class, pretrained_weights = MODELS[0]
        weight = pretrained_weights
        weight_dir = pretrained_weights

        weight_dir = "../Task2-Classification_of_Thesis/lmft/"
        weight = os.path.join(weight_dir, "checkpoint-23000")
        print(weight)
        self.tokenizer = tokenizer_class.from_pretrained(weight_dir)#, cache_dir="./cache")
        self.bert_model = model_class.from_pretrained(weight)#, cache_dir="./cache")

        self.padded_len = 60
        self.pad_id = self.tokenizer.encode("[PAD]")[0]
        print("pad_id", self.pad_id)

        self.estimate_clf = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                #nn.BatchNorm1d(self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.clf_dropout),
                nn.Linear(self.hidden_size // 2, num_classes),
                nn.Sigmoid()
                )


        self.clf2_linear1 = nn.Linear(self.hidden_size + self.num_classes + self.num_classes, hidden_size // 2)
        #self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.clf2_linear12 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear2 = nn.Linear(self.hidden_size + self.num_classes + self.num_classes + 1, hidden_size // 2)
        #self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.clf2_linear22 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear3 = nn.Linear(self.hidden_size + self.num_classes + self.num_classes + 2, hidden_size // 2)
        #self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.clf2_linear32 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear4 = nn.Linear(self.hidden_size + self.num_classes + self.num_classes + 3, hidden_size // 2)
        #self.bn4 = nn.BatchNorm1d(hidden_size // 2)
        self.clf2_linear42 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear5 = nn.Linear(self.hidden_size + self.num_classes + self.num_classes + 4, hidden_size // 2)
        #self.bn5 = nn.BatchNorm1d(hidden_size // 2)
        self.clf2_linear52 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear6 = nn.Linear(self.hidden_size + self.num_classes + self.num_classes + 5, hidden_size // 2)
        #self.bn6 = nn.BatchNorm1d(hidden_size // 2)
        self.clf2_linear62 = nn.Linear(hidden_size // 2, 1)
        self.Sigmoid = nn.Sigmoid()

        #self.relu = F.gelu
        #self.dpo = nn.Dropout(p=self.clf_dropout)

    def _pad_to_len(self, arr):
        if len(arr) > self.padded_len:
            arr = arr[:self.padded_len]
        while len(arr) < self.padded_len:
            arr.append(self.pad_id)
            
        return arr

    def submodel(self,former_batch, sentence_batch, after_batch):
        former_batch = former_batch.to("cuda")
        after_batch = after_batch.to("cuda")
        
        
        x1 = torch.cat((former_batch, sentence_batch, after_batch),1)
        x = self.clf2_linear1(x1)
        #x = self.bn1(x)
        x = F.relu(x)
        #x = self.dpo(x)
        x = self.clf2_linear12(x)
        x = self.Sigmoid(x)
        x2 = torch.cat((x1,x),1)
        y = self.clf2_linear2(x2)
        #y = self.bn2(y)
        y = F.relu(y)
        #y = self.dpo(y)
        y = self.clf2_linear22(y)
        y = self.Sigmoid(y)
        y2 = torch.cat((x2,y),1)
        i = self.clf2_linear3(y2)
        #i = self.bn3(i)
        i = F.relu(i)
        #i = self.dpo(i)
        i = self.clf2_linear32(i)
        i = self.Sigmoid(i)
        i2 = torch.cat((y2,i),1)
        j = self.clf2_linear4(i2)
        #j = self.bn4(j)
        j = F.relu(j)
        #j = self.dpo(j)
        j = self.clf2_linear42(j)
        j = self.Sigmoid(j)
        j2 = torch.cat((i2,j),1)
        k = self.clf2_linear5(j2)
        #k = self.bn5(k)
        k = F.relu(k)
        #k = self.dpo(k)
        k = self.clf2_linear52(k)
        k = self.Sigmoid(k)
        k2 = torch.cat((j2,k),1)
        m = self.clf2_linear6(k2)
        #m = self.bn6(m)
        m = F.relu(m)
        #m = self.dpo(m)
        m = self.clf2_linear62(m)
        m = self.Sigmoid(m)
        score = torch.cat((x,y,i,j,k,m),1)        

        return score

    def forward(self, batch):
        sentence_per_article = torch.IntTensor([len(article) for article in batch])
        max_sentence_per_article = torch.max(sentence_per_article).item()
        #print(sentence_per_article)
        #print(max_sentence_per_article)

        whole_sentence = []
        for article in batch :
            whole_sentence += article
        del batch
        torch.cuda.empty_cache()

        with torch.no_grad():
            input_id = torch.tensor([self._pad_to_len(self.tokenizer.encode(sent, add_special_tokens=True)) # add [CLS] [PAD]
                    for sent in whole_sentence]).to("cuda")
            attention_mask = (input_id != self.pad_id).float()
            #print(input_id.size()) # [128, 40]
            sentence_embedding = self.bert_model(input_id, attention_mask=attention_mask)[0][:,0,:].cuda()
            #print(sentence_embedding.size()) # [xxx, 758]

        estimate_result = self.estimate_clf(sentence_embedding).cuda()

        # transform
        article_start_index = [0]
        for i in range(len(sentence_per_article)):
            article_start_index.append(article_start_index[-1] + sentence_per_article[i].item())
            

        output = [[] for i in range(len(sentence_per_article))]
        if self.fp16:
            init = torch.zeros([self.num_classes]).cuda().type(torch.half)
        else:
            init = torch.zeros([self.num_classes]).cuda()
        for i in range(max_sentence_per_article):
            output_belong = []
            sentence_batch = []
            former_batch = []
            after_batch = []
            for j in range(len(sentence_per_article)):
                index = article_start_index[j] + i
                if index < article_start_index[j + 1]:
                    sentence_batch.append(sentence_embedding[article_start_index[j] + i])
                    if i == 0:
                        former_batch.append(init)
                    else:
                        former_batch.append(output[j][-1])
                    if index == article_start_index[j + 1] - 1:
                        after_batch.append(init)
                    else:
                        after_batch.append(estimate_result[article_start_index[j] + i + 1])
                    output_belong.append(j)

            former_batch = torch.stack(former_batch, dim=0)
            after_batch = torch.stack(after_batch, dim=0)
            if self.fp16:
                sentence_batch = torch.stack(sentence_batch, dim=0).type(torch.half)
            else:
                sentence_batch = torch.stack(sentence_batch, dim=0)

            
            score = self.submodel(former_batch, sentence_batch, after_batch)
            for i in range(len(score)):
                output[output_belong[i]].append(score[i][:])
        for i in range(len(output)):
            output[i] = torch.stack(output[i], dim=0)
        #print("output", output)

        return output
