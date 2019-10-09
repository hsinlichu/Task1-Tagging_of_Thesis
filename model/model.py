import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


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
                nn.Linear(hidden_size // 2, num_classes)
                )

    def forward(self, sentence):
        with torch.no_grad():
            sentence = self.embedding(sentence)
        #print(sentence.size()) # torch.Size([128, 30, 300])
        sentence_out, hidden = self.rnn(sentence)

        last_output = sentence_out[:,-1,:]

        score = self.clf(last_output)
        print(score.size()) # torch.Size([64, 37])
        return score
