import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        ######

        # 4.1 YOUR CODE HERE
        self.embed_layer = nn.Embedding.from_pretrained(torch.FloatTensor(vocab.vectors))
        self.fc1 = nn.Linear(embedding_dim, 1)

        ######

    def forward(self, x, lengths=None):

        ######

        # 4.2 YOUR CODE HERE

        x = self.embed_layer(torch.LongTensor(x))
        x = sum(x)/(lengths[0].type(torch.FloatTensor))
        x = self.fc1(x)
        x = torch.sigmoid(x)

        return x

        ######


class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()

        ######

        # 4.3 YOUR CODE HERE

        self.embed_layer = nn.Embedding.from_pretrained(torch.FloatTensor(vocab.vectors))
        self.GRULayer = nn.GRU(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(embedding_dim, 1)

        ######

    def forward(self, x, lengths=None):

        ######

        # 4.3 YOUR CODE HERE

        x = self.embed_layer(torch.LongTensor(x))
        # Comment out padded if for 7.3
        x = nn.utils.rnn.pack_padded_sequence(x, lengths)
        x = self.GRULayer(x)
        x = self.fc1(x[1].squeeze())
        x = torch.sigmoid(x)

        return x

        ######


class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()

        ######

        # 4.4 YOUR CODE HERE

        self.embed_layer = nn.Embedding.from_pretrained(torch.FloatTensor(vocab.vectors))
        self.conv1 = nn.Conv1d(embedding_dim, n_filters, filter_sizes[0])
        self.conv2 = nn.Conv1d(embedding_dim, n_filters, filter_sizes[1])
        self.fc1 = nn.Linear(embedding_dim, 1)

        ######

    def forward(self, x, lengths=None):
        ######

        # 4.4 YOUR CODE HERE

        x = self.embed_layer(torch.LongTensor(x))
        x1 = self.conv1(x.permute(1, 2, 0))
        x1 = F.relu(x1)
        #   L_out = (L_in=lengths - dilation=1 * (kernel_size=2 - 1) - 1 )/ stride=1 + 1
        #         = lengths - 1 - 1 + 1
        #         = lengths - 1
        x1 = nn.functional.max_pool1d(x1, int(lengths[0] - 1))

        x2 = self.conv2(x.permute(1, 2, 0))
        x2 = F.relu(x2)
        #   L_out = (L_in=lengths - dilation=1 * (kernel_size=4 - 1) - 1 )/ stride=1 + 1
        #         = lengths - 3 - 1 + 1
        #         = lengths - 3
        x2 = nn.functional.max_pool1d(x2, int(lengths[0]) - 3)

        x = self.fc1(torch.cat((x1, x2), 1).squeeze())
        x = torch.sigmoid(x)

        return x

        ######
