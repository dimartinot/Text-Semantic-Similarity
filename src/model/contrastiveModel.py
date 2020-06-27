import torch
import torch.nn.functional as F

import numpy as np

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

        euclidean_distance = torch.dist(output1, output2, p=2)
        loss_contrastive = torch.mean(1/2*(label) * torch.pow(euclidean_distance, 2) +
                                      1/2*(1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

class LstmNet(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim1=48, hidden_dim2=32, hidden_dim3=16):
        super(LstmNet, self).__init__()

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim1,
                            dropout=0.2,
                            batch_first=True, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(2*hidden_dim1, hidden_dim2, dropout=0.2, batch_first=True, bidirectional=True)
        self.lstm3 = torch.nn.LSTM(2*hidden_dim2 , hidden_dim3, dropout=0.2, batch_first=True, bidirectional=True)

        
    def forward(self, sequence):

        x, _ = self.lstm(sequence)
        
        x = torch.nn.ReLU()(x)
        
        x, _ = self.lstm2(x)

        x = torch.nn.ReLU()(x)

        _, (h, _) = self.lstm3(x)

        return h[-1,:,:]

class SiameseLSTM(torch.nn.Module):

    def __init__(self, embedding_dim):
        super(SiameseLSTM, self).__init__()

        self.net = LstmNet(embedding_dim)

        
    def forward(self, sequence1, sequence2):

        output1 = self.net(sequence1)
        output2 = self.net(sequence2)

        return output1, output2

class TripletLSTM(torch.nn.Module):

    def __init__(self, embedding_dim):
        super(TripletLSTM, self).__init__()

        self.net = LstmNet(embedding_dim)

    def forward(self, seq1, seq2, seq3):

        output1 = self.net(seq1)
        output2 = self.net(seq2)
        output3 = self.net(seq3)

        return output1, output2, output3

class QuadrupletLSTM(torch.nn.Module):

    def __init__(self, embedding_dim):
        super(QuadrupletLSTM, self).__init__()
        self.net = LstmNet(embedding_dim)

    def forward(self, seq1, seq2, seq3, seq4):

        output1 = self.net(seq1)
        output2 = self.net(seq2)
        output3 = self.net(seq3)
        output4 = self.net(seq4)

        return output1, output2, output3, output4