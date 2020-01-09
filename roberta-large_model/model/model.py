import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from transformers import *



class ThesisTaggingModelbert(BaseModel):
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
        #self.embedding = nn.Embedding(embedding.vectors.size(0), embedding.vectors.size(1))
        #self.embedding.weight = nn.Parameter(embedding.vectors)
        
        self.bertembedding = BertModel.from_pretrained('bert-base-uncased')
        

        self.rnn = nn.LSTM(input_size=self.dim_embeddings, hidden_size=self.hidden_size,
                num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True) # , dropout=rnn_dropout
        self.clf = nn.Sequential(
                nn.Linear(self.dim_embeddings, hidden_size // 2),
                #nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.clf_dropout),
                nn.Linear(hidden_size // 2, num_classes),
                nn.Sigmoid()
                )
        self.clf2_linear1 = nn.Linear(self.dim_embeddings,hidden_size)
        self.clf2_linear12 = nn.Linear(hidden_size,1)
        self.clf2_linear2 = nn.Linear(self.dim_embeddings+1,hidden_size)
        self.clf2_linear22 = nn.Linear(hidden_size,1)
        self.clf2_linear3 = nn.Linear(self.dim_embeddings+2,hidden_size)
        self.clf2_linear32 = nn.Linear(hidden_size,1)
        self.clf2_linear4 = nn.Linear(self.dim_embeddings+3,hidden_size)
        self.clf2_linear42 = nn.Linear(hidden_size,1)
        self.clf2_linear5 = nn.Linear(self.dim_embeddings+4,hidden_size)
        self.clf2_linear52 = nn.Linear(hidden_size,1)
        self.clf2_linear6 = nn.Linear(self.dim_embeddings+5,hidden_size)
        self.clf2_linear62 = nn.Linear(hidden_size,1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, sentence):
        #with torch.no_grad():
            #print(sentence)
            #print(sentence.size()) #torch.Size([128, 30])
            #sentence = self.embedding(sentence)
        sentence = self.bertembedding(sentence)[0] #(should be 128,30,1024)
        #print(sentence)
        #print(sentence.size()) # torch.Size([128, 30, 300])
        
        #sentence_out, hidden = self.rnn(sentence)
        #sentence = torch.mean(sentence,dim = 1)
        sentence = sentence[:,0,:]

        #last_output = sentence_out[:,-1,:]
        x = self.clf2_linear1(sentence)
        x = self.clf2_linear12(x)
        x = self.Sigmoid(x)
        x2 = torch.cat((sentence,x),1)
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
        #print(score.size())

        #score = self.clf(sentence)
        #print(score.size())
        #print(score)
        #print(score.size())
        return score




######################XLNET######################


class ThesisTaggingModelxlnet(BaseModel):
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
        
        self.XLembedding = XLNetModel.from_pretrained('xlnet-base-cased')
        #self.XLCF = XLNetForSequenceClassification.from_pretrained('bert-base-uncased')

        self.rnn = nn.LSTM(input_size=self.dim_embeddings, hidden_size=self.hidden_size,
                num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True) # , dropout=rnn_dropout
        self.clf = nn.Sequential(
                nn.Linear(self.dim_embeddings,hidden_size // 2),
                #nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.clf_dropout),
                nn.Linear(hidden_size // 2, num_classes),
                nn.Sigmoid()
                )
        
        


    def forward(self, sentence):
        with torch.no_grad():
            #print(sentence)
            #print(sentence.size()) #torch.Size([128, 30])
            #sentence = self.embedding(sentence)
            sentence = self.XLembedding(sentence)[0] #(should be 128,30,1024)
        #print(sentence)
        #print(sentence.size()) # torch.Size([128, 30, 300])
        sentence = sentence.mean()
        
        
        

        #last_output = sentence_out[:,-1,:]

        score = self.clf(sentence)
        
        #print(score)
        #print(score.size())
        return score
####################roBERTa######################


class ThesisTaggingModelroberta(BaseModel):
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
        #self.embedding = nn.Embedding(embedding.vectors.size(0), embedding.vectors.size(1))
        #self.embedding.weight = nn.Parameter(embedding.vectors)
        #(RobertaModel,    RobertaTokenizer,    'roberta-base')
        self.robertaembedding = RobertaModel.from_pretrained('roberta-base')
        

        self.rnn = nn.LSTM(input_size=self.dim_embeddings, hidden_size=self.hidden_size,
                num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True) # , dropout=rnn_dropout
        self.clf = nn.Sequential(
                nn.Linear(self.dim_embeddings, hidden_size // 2),
                #nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.clf_dropout),
                nn.Linear(hidden_size // 2, num_classes),
                nn.Sigmoid()
                )
        self.clf2_linear1 = nn.Linear(self.dim_embeddings,hidden_size)
        self.clf2_linear11 = nn.Linear(hidden_size,hidden_size//2)
        self.clf2_linear12 = nn.Linear(hidden_size//2,1)
        self.clf2_linear2 = nn.Linear(self.dim_embeddings+1,hidden_size)
        self.clf2_linear21 = nn.Linear(hidden_size,hidden_size//2)
        self.clf2_linear22 = nn.Linear(hidden_size//2,1)
        self.clf2_linear3 = nn.Linear(self.dim_embeddings+2,hidden_size)
        self.clf2_linear31 = nn.Linear(hidden_size,hidden_size//2)
        self.clf2_linear32 = nn.Linear(hidden_size//2,1)
        self.clf2_linear4 = nn.Linear(self.dim_embeddings+3,hidden_size)
        self.clf2_linear41 = nn.Linear(hidden_size,hidden_size//2)
        self.clf2_linear42 = nn.Linear(hidden_size//2,1)
        self.clf2_linear5 = nn.Linear(self.dim_embeddings+4,hidden_size)
        self.clf2_linear51 = nn.Linear(hidden_size,hidden_size//2)
        self.clf2_linear52 = nn.Linear(hidden_size//2,1)
        self.clf2_linear6 = nn.Linear(self.dim_embeddings+5,hidden_size)
        self.clf2_linear61 = nn.Linear(hidden_size,hidden_size//2)
        self.clf2_linear62 = nn.Linear(hidden_size//2,1)
        self.Sigmoid = nn.Sigmoid()
        self.Relu = nn.ReLU()

    def forward(self, sentence):
        #with torch.no_grad():
            #print(sentence)
            #print(sentence.size()) #torch.Size([128, 30])
            #sentence = self.embedding(sentence)
        sentence = self.robertaembedding(sentence)[0] #(should be 128,30,1024)
        #print(sentence)
        #print(sentence.size()) # torch.Size([128, 30, 300])
        
        #sentence_out, hidden = self.rnn(sentence)
        #sentence = torch.mean(sentence,dim = 1)
        sentence = sentence[:,0,:]

        #last_output = sentence_out[:,-1,:]
        x = self.clf2_linear1(sentence)
        x = self.Relu(x)
        x = self.clf2_linear11(x)
        x = self.Relu(x)
        x = self.clf2_linear12(x)
        x = self.Sigmoid(x)
        x2 = torch.cat((sentence,x),1)
        y = self.clf2_linear2(x2)
        y = self.Relu(y)
        y = self.clf2_linear21(y)
        y = self.Relu(y)
        y = self.clf2_linear22(y)
        y = self.Sigmoid(y)
        y2 = torch.cat((x2,y),1)
        i = self.clf2_linear3(y2)
        i = self.Relu(i)
        i = self.clf2_linear31(i)
        i = self.Relu(i)
        i = self.clf2_linear32(i)
        i = self.Sigmoid(i)
        i2 = torch.cat((y2,i),1)
        j = self.clf2_linear4(i2)
        j = self.Relu(j)
        j = self.clf2_linear41(j)
        j = self.Relu(j)
        j = self.clf2_linear42(j)
        j = self.Sigmoid(j)
        j2 = torch.cat((i2,j),1)
        k = self.clf2_linear5(j2)
        k = self.Relu(k)
        k = self.clf2_linear51(k)
        k = self.Relu(k)
        k = self.clf2_linear52(k)
        k = self.Sigmoid(k)
        k2 = torch.cat((j2,k),1)
        m = self.clf2_linear6(k2)
        m = self.Relu(m)
        m = self.clf2_linear61(m)
        m = self.Relu(m)
        m = self.clf2_linear62(m)
        m = self.Sigmoid(m)
        
        score = torch.cat((x,y,i,j,k,m),1)
        #print(score.size())

        # score = self.clf(sentence)
        #print(score.size())
        #print(score)
        #print(score.size())
        return score




####################roberta - base######################



class ThesisTaggingModelrobertabase(BaseModel):
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
        #self.embedding = nn.Embedding(embedding.vectors.size(0), embedding.vectors.size(1))
        #self.embedding.weight = nn.Parameter(embedding.vectors)
        #(RobertaModel,    RobertaTokenizer,    'roberta-base')
        self.robertaembedding = RobertaModel.from_pretrained('roberta-base')
        

        self.rnn = nn.LSTM(input_size=self.dim_embeddings, hidden_size=self.hidden_size,
                num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True) # , dropout=rnn_dropout
        self.clf = nn.Sequential(
                nn.Linear(self.dim_embeddings, hidden_size // 2),
                #nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.clf_dropout),
                nn.Linear(hidden_size // 2, num_classes),
                nn.Sigmoid()
                )
        self.clf2_linear1 = nn.Linear(self.dim_embeddings,hidden_size)
        self.clf2_linear11 = nn.Linear(hidden_size,hidden_size//2)
        self.clf2_linear12 = nn.Linear(hidden_size//2,1)
        self.clf2_linear2 = nn.Linear(self.dim_embeddings+1,hidden_size)
        self.clf2_linear21 = nn.Linear(hidden_size,hidden_size//2)
        self.clf2_linear22 = nn.Linear(hidden_size//2,1)
        self.clf2_linear3 = nn.Linear(self.dim_embeddings+2,hidden_size)
        self.clf2_linear31 = nn.Linear(hidden_size,hidden_size//2)
        self.clf2_linear32 = nn.Linear(hidden_size//2,1)
        self.clf2_linear4 = nn.Linear(self.dim_embeddings+3,hidden_size)
        self.clf2_linear41 = nn.Linear(hidden_size,hidden_size//2)
        self.clf2_linear42 = nn.Linear(hidden_size//2,1)
        self.clf2_linear5 = nn.Linear(self.dim_embeddings+4,hidden_size)
        self.clf2_linear51 = nn.Linear(hidden_size,hidden_size//2)
        self.clf2_linear52 = nn.Linear(hidden_size//2,1)
        self.clf2_linear6 = nn.Linear(self.dim_embeddings+5,hidden_size)
        self.clf2_linear61 = nn.Linear(hidden_size,hidden_size//2)
        self.clf2_linear62 = nn.Linear(hidden_size//2,1)
        self.Sigmoid = nn.Sigmoid()
        self.Relu = nn.ReLU()
        #self.clf31 = nn.Linear(12,50)
        #self.clf32 = nn.Linear(50,25)
        #self.clf33 = nn.Linear(25,12)
        #self.clf34 = nn.Linear(12,6)

    def forward(self, sentence):
        with torch.no_grad():
            #print(sentence)
            #print(sentence.size()) #torch.Size([128, 30])
            #sentence = self.embedding(sentence)
            sentence = self.robertaembedding(sentence)[0] #(should be 128,30,1024)
            #print(sentence)
            #print(sentence.size()) # torch.Size([128, 30, 300])
        
            #sentence_out, hidden = self.rnn(sentence)
            #sentence = torch.mean(sentence,dim = 1)
            sentence = sentence[:,0,:]

        #last_output = sentence_out[:,-1,:]
            x = self.clf2_linear1(sentence)
            x = self.Relu(x)
            x = self.clf2_linear11(x)
            x = self.Relu(x)
            x = self.clf2_linear12(x)
            x = self.Sigmoid(x)
            x2 = torch.cat((sentence,x),1)
            y = self.clf2_linear2(x2)
            y = self.Relu(y)
            y = self.clf2_linear21(y)
            y = self.Relu(y)
            y = self.clf2_linear22(y)
            y = self.Sigmoid(y)
            y2 = torch.cat((x2,y),1)
            i = self.clf2_linear3(y2)
            i = self.Relu(i)
            i = self.clf2_linear31(i)
            i = self.Relu(i)
            i = self.clf2_linear32(i)
            i = self.Sigmoid(i)
            i2 = torch.cat((y2,i),1)
            j = self.clf2_linear4(i2)
            j = self.Relu(j)
            j = self.clf2_linear41(j)
            j = self.Relu(j)
            j = self.clf2_linear42(j)
            j = self.Sigmoid(j)
            j2 = torch.cat((i2,j),1)
            k = self.clf2_linear5(j2)
            k = self.Relu(k)
            k = self.clf2_linear51(k)
            k = self.Relu(k)
            k = self.clf2_linear52(k)
            k = self.Sigmoid(k)
            k2 = torch.cat((j2,k),1)
            m = self.clf2_linear6(k2)
            m = self.Relu(m)
            m = self.clf2_linear61(m)
            m = self.Relu(m)
            m = self.clf2_linear62(m)
            m = self.Sigmoid(m)
        
        score = torch.cat((x,y,i,j,k,m),1)
        #score = torch.cat((score,))
        #score = = self.clf31
        #print(score.size())

        # score = self.clf(sentence)
        #print(score.size())
        #print(score)
        #print(score.size())
        return score


##################################CASCADE##########################


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
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        #logging.info("Embedding size: ({},{})".format(embedding.size(0),embedding.size(1)))
        #self.embedding = nn.Embedding(embedding.vectors.size(0), embedding.vectors.size(1))
        #self.embedding.weight = nn.Parameter(embedding.vectors)
        self.robertaembedding = RobertaModel.from_pretrained('roberta-large')

        #self.rnn = nn.LSTM(input_size=self.dim_embeddings, hidden_size=self.hidden_size,
        #        num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True) # , dropout=rnn_dropout

        
        self.clf2_linear1 = nn.Linear(self.dim_embeddings  , hidden_size // 2)
        self.clf2_linear12 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear2 = nn.Linear(self.dim_embeddings + 1, hidden_size // 2)
        self.clf2_linear22 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear3 = nn.Linear(self.dim_embeddings  + 2, hidden_size // 2)
        self.clf2_linear32 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear4 = nn.Linear(self.dim_embeddings + 3, hidden_size // 2)
        self.clf2_linear42 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear5 = nn.Linear(self.dim_embeddings+ 4, hidden_size // 2)
        self.clf2_linear52 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear6 = nn.Linear(self.dim_embeddings + 5, hidden_size // 2)
        self.clf2_linear62 = nn.Linear(hidden_size // 2, 1)
        self.clf2_catpre =  nn.Linear(12,12)
        self.clf2_catpre2 = nn.Linear(12,6)
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        #self.gelu = F.gelu()
        self.clf1 = nn.Linear(self.dim_embeddings, hidden_size // 2)
        self.clf1_2 = nn.Linear(hidden_size // 2, num_classes)

        self.clf31 = nn.Linear(hidden_size ,hidden_size//2)
        self.clf32 = nn.Linear(hidden_size//2,hidden_size//4)
        self.clf33 = nn.Linear(hidden_size//4,6)
        
        self.clf41 = nn.Linear(18,120)
        self.clf42 = nn.Linear(120,60)
        self.clf43 = nn.Linear(60,30)
        self.clf44 = nn.Linear(30,6)

    def submodel(self, former_output,init2  ,sentence):
        #print("former_output", former_output.size())
        #print("sentence", sentence.size())
        #sentence = sentence.to("cuda")
        #with torch.no_grad():
        ain = (sentence != self.tokenizer.encode('<pad>')[0]).float()
        #print(sentence)
        #print(ain)
        #print(ain.size())
        #sentence_out = self.robertaembedding(sentence,attention_mask = ain)[0]
        former_output = former_output.to("cuda")
        sentence = sentence.to("cuda")
        init2 = init2.to("cuda")
        with torch.no_grad():
            #print(sentence.size()) # torch.Size([128, 30, 300])
            #sentence_out, hidden = self.rnn(sentence)
            sentence_out = self.robertaembedding(sentence,attention_mask = ain)[0]
            last_output = sentence_out[:,0,:]
        
        
        
            #x1 = torch.cat((former_output, last_output),1)
            #x = self.clf1(last_output)
            #x = self.relu(x)
            #x = self.clf1_2(x)
            x1 = last_output
            x = self.clf2_linear1(x1)
            x = F.gelu(x)
            x = self.clf2_linear12(x)
            x = self.Sigmoid(x)
            x2 = torch.cat((x1,x),1)
            y = self.clf2_linear2(x2)
            y = F.gelu(y)
            y = self.clf2_linear22(y)
            y = self.Sigmoid(y)
            y2 = torch.cat((x2,y),1)
            i = self.clf2_linear3(y2)
            i = F.gelu(i)
            i = self.clf2_linear32(i)
            i = self.Sigmoid(i)
            i2 = torch.cat((y2,i),1)
            j = self.clf2_linear4(i2)
            j = F.gelu(j)
            j = self.clf2_linear42(j)
            j = self.Sigmoid(j)
            j2 = torch.cat((i2,j),1)
            k = self.clf2_linear5(j2)
            k = F.gelu(k)
            k = self.clf2_linear52(k)
            k = self.Sigmoid(k)
            k2 = torch.cat((j2,k),1)
            m = self.clf2_linear6(k2)
            m = F.gelu(m)
            m = self.clf2_linear62(m)
            m = self.Sigmoid(m)
            score = torch.cat((x,y,i,j,k,m),1)
            
            if not torch.equal(sentence, init2):
                ain = (init2 != self.tokenizer.encode('<pad>')[0]).float()
                sentence_out = self.robertaembedding(init2,attention_mask = ain)[0]
                #print(sentence.size()) # torch.Size([128, 30, 300])
                #sentence_out, hidden = self.rnn(sentence)
                last_output = sentence_out[:,0,:]
                #x1 = torch.cat((former_output, last_output),1)
                #x = self.clf1(last_output)
                #x = self.relu(x)
                #x = self.clf1_2(x)
                x1 = last_output
                x = self.clf2_linear1(x1)
                x = F.gelu(x)
                x = self.clf2_linear12(x)
                x = self.Sigmoid(x)
                x2 = torch.cat((x1,x),1)
                y = self.clf2_linear2(x2)
                y = F.gelu(y)
                y = self.clf2_linear22(y)
                y = self.Sigmoid(y)
                y2 = torch.cat((x2,y),1)
                i = self.clf2_linear3(y2)
                i = F.gelu(i)
                i = self.clf2_linear32(i)
                i = self.Sigmoid(i)
                i2 = torch.cat((y2,i),1)
                j = self.clf2_linear4(i2)
                j = F.gelu(j)
                j = self.clf2_linear42(j)
                j = self.Sigmoid(j)
                j2 = torch.cat((i2,j),1)
                k = self.clf2_linear5(j2)
                k = F.gelu(k)
                k = self.clf2_linear52(k)
                k = self.Sigmoid(k)
                k2 = torch.cat((j2,k),1)
                m = self.clf2_linear6(k2)
                m = F.gelu(m)
                m = self.clf2_linear62(m)
                m = self.Sigmoid(m)
                score2 = torch.cat((x,y,i,j,k,m),1)
            else :
                score2 = torch.zeros([ sentence.size()[0], 6]).to('cuda')
            
        #score2 = init2
        score = torch.cat((former_output,score,score2),1)
        #print(former_output)
        #print(init2)
        #print(score)
        s = self.clf41(score)
        s = F.gelu(s)
        s = self.clf42(s)
        s = F.gelu(s)
        s = self.clf43(s)
        #s = self.relu(s)
        score = self.clf44(s)
        
        #score = torch.cat((former_output,score),1)
        #score = self.clf2_catpre(score)
        #score = self.relu(score)
        #score = self.clf2_catpre2(score)
        score = self.Sigmoid(score)

        return score

    def forward(self, batch):
        sentence_per_article = torch.IntTensor([article.size(0) for article in batch])
        max_sentence_per_article = torch.max(sentence_per_article).item()
        #print(sentence_per_article)
        #print(max_sentence_per_article)
        #print(sentence_per_article[0])
        #print()
        for i in range(len(batch)):
            batch[i] = F.pad(batch[i], (0, 0, 0, max_sentence_per_article - batch[i].size()[0]), "constant", 0)
        batch = torch.stack(batch)#.to("cuda")

        output = []
        init = torch.zeros([ batch.size()[0], 6])
        former_output = init
        former_output2 = torch.zeros([ batch.size()[0], 6])
        #init2 = torch.zeros([ batch.size()[0], 6])
        for i in range(max_sentence_per_article):
            if i != max_sentence_per_article-1:
                #score = self.submodel( former_output,former_output2, batch[:, i, :].to("cuda"))
                score = self.submodel( former_output,batch[:,i+1,:].to("cuda"), batch[:, i, :].to("cuda"))
            else:
                #score = self.submodel( former_output,former_output2, batch[:, i, :].to("cuda"))
                score = self.submodel( former_output,batch[:,i,:].to("cuda"), batch[:, i, :].to("cuda"))
            output.append(score)
            #init2 = former_output
            former_output2 = former_output #####
            former_output = score

        output = torch.stack(output, dim=1)        
        #print(output.size())
        ret = []
        for i in range(output.size(0)):
            ret.append(output[i][:sentence_per_article[i]])
        #print(output[0])
        #print(output[0][:sentence_per_article[0]])
        #print(ret.size())
        return ret
'''

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
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        #logging.info("Embedding size: ({},{})".format(embedding.size(0),embedding.size(1)))
        #self.embedding = nn.Embedding(embedding.vectors.size(0), embedding.vectors.size(1))
        #self.embedding.weight = nn.Parameter(embedding.vectors)
        self.robertaembedding = RobertaModel.from_pretrained('roberta-base')#roberta-large

        #self.rnn = nn.LSTM(input_size=self.dim_embeddings, hidden_size=self.hidden_size,
        #        num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True) # , dropout=rnn_dropout

        
        self.clf2_linear1 = nn.Linear(self.dim_embeddings  , hidden_size // 2)
        self.clf2_linear12 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear2 = nn.Linear(self.dim_embeddings + 1, hidden_size // 2)
        self.clf2_linear22 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear3 = nn.Linear(self.dim_embeddings  + 2, hidden_size // 2)
        self.clf2_linear32 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear4 = nn.Linear(self.dim_embeddings + 3, hidden_size // 2)
        self.clf2_linear42 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear5 = nn.Linear(self.dim_embeddings+ 4, hidden_size // 2)
        self.clf2_linear52 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear6 = nn.Linear(self.dim_embeddings + 5, hidden_size // 2)
        self.clf2_linear62 = nn.Linear(hidden_size // 2, 1)
        
        self.clf1 = nn.Linear(self.dim_embeddings, hidden_size // 2)
        self.clf1_2 = nn.Linear(hidden_size // 2, num_classes)

        self.clf2_catpre =  nn.Linear(12,12)
        self.clf2_catpre2 = nn.Linear(12,6)
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        #self.gelu = F.gelu() 
        self.clf31 = nn.Linear(hidden_size ,hidden_size//2)
        self.clf32 = nn.Linear(hidden_size//2,hidden_size//4)
        self.clf33 = nn.Linear(hidden_size//4,6)
        #self.clf34 = nn.Linear(15,6)

        self.clf41 = nn.Linear(18,120)
        self.clf42 = nn.Linear(120,60)
        self.clf43 = nn.Linear(60,30)
        self.clf44 = nn.Linear(30,6)

        #self.clf51 = nn.Linear(30,120)
        #self.clf52 = nn.Linear(120,60)
        #self.clf53 = nn.Linear(60,30)
        #self.clf54 = nn.Linear(30,6)

        #self.clf61 = nn.Linear(42,120)
        #self.clf62 = nn.Linear(120,60)
        #self.clf63 = nn.Linear(60,30)
        #self.clf64 = nn.Linear(30,6)


    def submodel(self, former_output,  sentence):
        #print("former_output", former_output.size())
        #print("sentence", sentence.size())
        #former_output = former_output.to("cuda")
        sentence = sentence.to("cuda")
        #with torch.no_grad():
        ain = (sentence != self.tokenizer.encode('<pad>')[0]).float()

        sentence_out = self.robertaembedding(sentence,attention_mask = ain)[0]
            #print(sentence.size()) # torch.Size([128, 30, 300])
            #sentence_out, hidden = self.rnn(sentence)
        last_output = sentence_out[:,0,:]
        

        x1 = last_output
        x = self.clf2_linear1(x1)
        x = F.gelu(x)
        x = self.clf2_linear12(x)
        x = self.Sigmoid(x)
        x2 = torch.cat((x1,x),1)
        y = self.clf2_linear2(x2)
        y = F.gelu(y)
        y = self.clf2_linear22(y)
        y = self.Sigmoid(y)
        y2 = torch.cat((x2,y),1)
        i = self.clf2_linear3(y2)
        i = F.gelu(i)
        i = self.clf2_linear32(i)
        i = self.Sigmoid(i)
        i2 = torch.cat((y2,i),1)
        j = self.clf2_linear4(i2)
        j = F.gelu(j)
        j = self.clf2_linear42(j)
        j = self.Sigmoid(j)
        j2 = torch.cat((i2,j),1)
        k = self.clf2_linear5(j2)
        k = F.gelu(k)
        k = self.clf2_linear52(k)
        k = self.Sigmoid(k)
        k2 = torch.cat((j2,k),1)
        m = self.clf2_linear6(k2)
        m = F.gelu(m)
        m = self.clf2_linear62(m)
        m = self.Sigmoid(m)
        score = torch.cat((x,y,i,j,k,m),1)       


        return score

    def forward(self, batch):

        self.dataset = []
        j = 0
        for i in range(len(batch)):
            if i == 0:
                j = batch[i]
            else:
                j = torch.cat((j,batch[i]),0)

        score = self.submodel( 0, j.to("cuda"))


        return score
'''


'''
#######backward2 backward2######

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
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        #logging.info("Embedding size: ({},{})".format(embedding.size(0),embedding.size(1)))
        #self.embedding = nn.Embedding(embedding.vectors.size(0), embedding.vectors.size(1))
        #self.embedding.weight = nn.Parameter(embedding.vectors)
        self.robertaembedding = RobertaModel.from_pretrained('roberta-large')

        #self.rnn = nn.LSTM(input_size=self.dim_embeddings, hidden_size=self.hidden_size,
        #        num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True) # , dropout=rnn_dropout

        
        self.clf2_linear1 = nn.Linear(self.dim_embeddings  , hidden_size // 2)
        self.clf2_linear12 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear2 = nn.Linear(self.dim_embeddings + 1, hidden_size // 2)
        self.clf2_linear22 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear3 = nn.Linear(self.dim_embeddings  + 2, hidden_size // 2)
        self.clf2_linear32 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear4 = nn.Linear(self.dim_embeddings + 3, hidden_size // 2)
        self.clf2_linear42 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear5 = nn.Linear(self.dim_embeddings+ 4, hidden_size // 2)
        self.clf2_linear52 = nn.Linear(hidden_size // 2, 1)
        self.clf2_linear6 = nn.Linear(self.dim_embeddings + 5, hidden_size // 2)
        self.clf2_linear62 = nn.Linear(hidden_size // 2, 1)
        self.clf2_catpre =  nn.Linear(12,12)
        self.clf2_catpre2 = nn.Linear(12,6)
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        #self.gelu = F.gelu()
        self.clf1 = nn.Linear(self.dim_embeddings, hidden_size // 2)
        self.clf1_2 = nn.Linear(hidden_size // 2, num_classes)

        self.clf31 = nn.Linear(hidden_size ,hidden_size//2)
        self.clf32 = nn.Linear(hidden_size//2,hidden_size//4)
        self.clf33 = nn.Linear(hidden_size//4,6)
        
        self.clf41 = nn.Linear(18,120)
        self.clf42 = nn.Linear(120,60)
        self.clf43 = nn.Linear(60,30)
        self.clf44 = nn.Linear(30,6)
        self.clf51 = nn.Linear(30,120)
        self.clf52 = nn.Linear(120,60)
        self.clf53 = nn.Linear(60,30)
        self.clf54 = nn.Linear(30,6)
        self.clf61 = nn.Linear(42,120)
        self.clf62 = nn.Linear(120,60)
        self.clf63 = nn.Linear(60,30)
        self.clf64 = nn.Linear(30,6)


    def submodel(self, former_output,init2  ,sentence):
        #print("former_output", former_output.size())
        #print("sentence", sentence.size())
        #sentence = sentence.to("cuda")
        #with torch.no_grad():
        ain = (sentence != self.tokenizer.encode('<pad>')[0]).float()
        #print(sentence)
        #print(ain)
        #print(ain.size())
        #sentence_out = self.robertaembedding(sentence,attention_mask = ain)[0]
        former_output = former_output.to("cuda")
        sentence = sentence.to("cuda")
        init2 = init2.to("cuda")
        with torch.no_grad():
            #print(sentence.size()) # torch.Size([128, 30, 300])
            #sentence_out, hidden = self.rnn(sentence)
            sentence_out = self.robertaembedding(sentence,attention_mask = ain)[0]
            last_output = sentence_out[:,0,:]
        
        
        
            #x1 = torch.cat((former_output, last_output),1)
            #x = self.clf1(last_output)
            #x = self.relu(x)
            #x = self.clf1_2(x)
            x1 = last_output
            x = self.clf2_linear1(x1)
            x = F.gelu(x)
            x = self.clf2_linear12(x)
            x = self.Sigmoid(x)
            x2 = torch.cat((x1,x),1)
            y = self.clf2_linear2(x2)
            y = F.gelu(y)
            y = self.clf2_linear22(y)
            y = self.Sigmoid(y)
            y2 = torch.cat((x2,y),1)
            i = self.clf2_linear3(y2)
            i = F.gelu(i)
            i = self.clf2_linear32(i)
            i = self.Sigmoid(i)
            i2 = torch.cat((y2,i),1)
            j = self.clf2_linear4(i2)
            j = F.gelu(j)
            j = self.clf2_linear42(j)
            j = self.Sigmoid(j)
            j2 = torch.cat((i2,j),1)
            k = self.clf2_linear5(j2)
            k = F.gelu(k)
            k = self.clf2_linear52(k)
            k = self.Sigmoid(k)
            k2 = torch.cat((j2,k),1)
            m = self.clf2_linear6(k2)
            m = F.gelu(m)
            m = self.clf2_linear62(m)
            m = self.Sigmoid(m)
            score = torch.cat((x,y,i,j,k,m),1)
            
            if not torch.equal(sentence, init2):
                ain = (init2 != self.tokenizer.encode('<pad>')[0]).float()
                sentence_out = self.robertaembedding(init2,attention_mask = ain)[0]
                #print(sentence.size()) # torch.Size([128, 30, 300])
                #sentence_out, hidden = self.rnn(sentence)
                last_output = sentence_out[:,0,:]
                #x1 = torch.cat((former_output, last_output),1)
                #x = self.clf1(last_output)
                #x = self.relu(x)
                #x = self.clf1_2(x)
                x1 = last_output
                x = self.clf2_linear1(x1)
                x = F.gelu(x)
                x = self.clf2_linear12(x)
                x = self.Sigmoid(x)
                x2 = torch.cat((x1,x),1)
                y = self.clf2_linear2(x2)
                y = F.gelu(y)
                y = self.clf2_linear22(y)
                y = self.Sigmoid(y)
                y2 = torch.cat((x2,y),1)
                i = self.clf2_linear3(y2)
                i = F.gelu(i)
                i = self.clf2_linear32(i)
                i = self.Sigmoid(i)
                i2 = torch.cat((y2,i),1)
                j = self.clf2_linear4(i2)
                j = F.gelu(j)
                j = self.clf2_linear42(j)
                j = self.Sigmoid(j)
                j2 = torch.cat((i2,j),1)
                k = self.clf2_linear5(j2)
                k = F.gelu(k)
                k = self.clf2_linear52(k)
                k = self.Sigmoid(k)
                k2 = torch.cat((j2,k),1)
                m = self.clf2_linear6(k2)
                m = F.gelu(m)
                m = self.clf2_linear62(m)
                m = self.Sigmoid(m)
                score2 = torch.cat((x,y,i,j,k,m),1)
            else :
                score2 = torch.zeros([ sentence.size()[0], 6]).to('cuda')
            
            if not torch.equal(sentence, former_output):
                ain = (former_output != self.tokenizer.encode('<pad>')[0]).float()
                sentence_out = self.robertaembedding(former_output,attention_mask = ain)[0]
                #print(sentence.size()) # torch.Size([128, 30, 300])
                #sentence_out, hidden = self.rnn(sentence)
                last_output = sentence_out[:,0,:]
                #x1 = torch.cat((former_output, last_output),1)
                #x = self.clf1(last_output)
                #x = self.relu(x)
                #x = self.clf1_2(x)
                x1 = last_output
                x = self.clf2_linear1(x1)
                x = F.gelu(x)
                x = self.clf2_linear12(x)
                x = self.Sigmoid(x)
                x2 = torch.cat((x1,x),1)
                y = self.clf2_linear2(x2)
                y = F.gelu(y)
                y = self.clf2_linear22(y)
                y = self.Sigmoid(y)
                y2 = torch.cat((x2,y),1)
                i = self.clf2_linear3(y2)
                i = F.gelu(i)
                i = self.clf2_linear32(i)
                i = self.Sigmoid(i)
                i2 = torch.cat((y2,i),1)
                j = self.clf2_linear4(i2)
                j = F.gelu(j)
                j = self.clf2_linear42(j)
                j = self.Sigmoid(j)
                j2 = torch.cat((i2,j),1)
                k = self.clf2_linear5(j2)
                k = F.gelu(k)
                k = self.clf2_linear52(k)
                k = self.Sigmoid(k)
                k2 = torch.cat((j2,k),1)
                m = self.clf2_linear6(k2)
                m = F.gelu(m)
                m = self.clf2_linear62(m)
                m = self.Sigmoid(m)
                score3 = torch.cat((x,y,i,j,k,m),1)
            else :
                score3 = torch.zeros([ sentence.size()[0], 6]).to('cuda')
            
        ####score2 = init2
        score = torch.cat((score3,score,score2),1)
        #print(former_output)
        #print(init2)
        #print(score)
        s = self.clf41(score)
        s = F.gelu(s)
        s = self.clf42(s)
        s = F.gelu(s)
        s = self.clf43(s)
        #s = self.relu(s)
        score = self.clf44(s)
        
        #score = torch.cat((former_output,score),1)
        #score = self.clf2_catpre(score)
        #score = self.relu(score)
        #score = self.clf2_catpre2(score)
        score = self.Sigmoid(score)

        return score

    def forward(self, batch):
        sentence_per_article = torch.IntTensor([article.size(0) for article in batch])
        max_sentence_per_article = torch.max(sentence_per_article).item()
        #print(sentence_per_article)
        #print(max_sentence_per_article)
        #print(sentence_per_article[0])
        #print()
        for i in range(len(batch)):
            batch[i] = F.pad(batch[i], (0, 0, 0, max_sentence_per_article - batch[i].size()[0]), "constant", 0)
        batch = torch.stack(batch)#.to("cuda")

        output = []
        init = torch.zeros([ batch.size()[0], 6])
        former_output = init
        former_output2 = torch.zeros([ batch.size()[0], 6])
        #init2 = torch.zeros([ batch.size()[0], 6])
        for i in range(max_sentence_per_article):
            if i == max_sentence_per_article-1:
                #score = self.submodel( former_output,former_output2, batch[:, i, :].to("cuda"))
                score = self.submodel( batch[:,i,:].to("cuda"),batch[:,i,:].to("cuda"), batch[:, i, :].to("cuda"))
            elif i == max_sentence_per_article-2:
                #score = self.submodel( former_output,former_output2, batch[:, i, :].to("cuda"))
                score = self.submodel( batch[:,i+1,:].to("cuda"),batch[:,i+1,:].to("cuda"), batch[:, i, :].to("cuda"))
            else:
                #score = self.submodel( former_output,former_output2, batch[:, i, :].to("cuda"))
                score = self.submodel( batch[:,i+2,:].to("cuda"),batch[:,i+1,:].to("cuda"), batch[:, i, :].to("cuda"))
                #score = self.submodel( batch[:,i,:].to("cuda"),batch[:,i,:].to("cuda"), batch[:, i, :].to("cuda"))
            output.append(score)
            #init2 = former_output
            former_output2 = former_output #####
            former_output = score

        output = torch.stack(output, dim=1)        
        #print(output.size())
        ret = []
        for i in range(output.size(0)):
            ret.append(output[i][:sentence_per_article[i]])
        #print(output[0])
        #print(output[0][:sentence_per_article[0]])
        #print(ret.size())
        return ret
'''
