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
        
class TextSimilarityLSTM(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim=30, fc1_size = 1024, fc2_size=256, fc3_size = 64):
        super(TextSimilarityLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, 2,
                            dropout=0.15 + np.random.rand() * 0.25,
                            batch_first=True, bidirectional=False)

        
    def forward(self, sentence1, sentence2):

        output1, (h_n1, c_n1) = self.lstm(sentence1)
        output2, (h_n2, c_n2)= self.lstm(sentence2)
        

        x1 = h_n1[-1,:,:]
        x2 = h_n2[-1,:,:]
                
        return x1, x2


class SiameseContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(SiameseContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, euclidean_distance, label):

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

class TextSimilarityDeepSiameseLSTM(torch.nn.Module):

    def __init__(self, embedding_dim):
        super(TextSimilarityDeepSiameseLSTM, self).__init__()

        self.net = LstmNet(embedding_dim)

        
    def forward(self, sequence1, sequence2):

        output1 = self.net(sequence1)
        output2 = self.net(sequence2)

        dist = torch.sum(torch.abs(output1 - output2), dim=1, keepdim=True)
        

        return dist


class DeepLstmNet(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim1=48, hidden_dim2=48):
        super(DeepLstmNet, self).__init__()

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.blstm = torch.nn.LSTM(embedding_dim, hidden_dim1, 2, dropout=0.2,
                            batch_first=True, bidirectional=True)
        self.blstm2 = torch.nn.LSTM(hidden_dim1*2, hidden_dim2, 2, dropout=0.2, batch_first=True, bidirectional=True)

        self.drop = torch.nn.Dropout(p=0.4)

        # *2 because bidirectional
        self.linear = torch.nn.Linear(hidden_dim2*2, 128)



    def forward(self, sequence):

        # output shape: (seq_len, batch, num_directions * hidden_size)
        output1, _ = self.blstm(sequence)
        output2, _ = self.blstm2(output1)

        m = torch.mean(
            output2, dim=1
        )

        out = self.linear(self.drop(m))

        return out


class SiameseRecurrentNetwork(torch.nn.Module):

    def __init__(self, embedding_dim):
        super(SiameseRecurrentNetwork, self).__init__()

        self.net = DeepLstmNet(embedding_dim)
        
    def forward(self, sequence1, sequence2):

        output1 = self.net(sequence1)
        output2 = self.net(sequence2)

        return output1, output2


class CosineLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(CosineLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label, c=1e-6):
        # Inspired by https://github.com/Kajiyu/MANNs/blob/master/models/utils.py

        out1_norm = torch.norm(out1, p=2, dim=1, keepdim=True) + c
        out2_norm = torch.norm(out2, p=2, dim=1, keepdim=True) + c

        dot = (out1*out2).sum(-1)

        energy = dot/(out1_norm*out2_norm).sum(-1)
    # apply_dict(locals())

        #print(energy.size())
        #d

        loss_contrastive = torch.mean(
            label * (1 / 4) * torch.pow(1 - energy, 2)
            + (1-label) * torch.where(energy < self.margin, torch.pow(energy, 2), torch.zeros_like(energy))
        )

        return loss_contrastive