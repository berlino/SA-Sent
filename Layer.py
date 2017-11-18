from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import pdb
import cPickle
import numpy as np
import math
from util import *

def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal(weight_)

class MLSTM(nn.Module):
    def __init__(self, config):
        super(MLSTM, self).__init__()
        self.config = config

        self.rnn = nn.LSTM(config.embed_dim + config.mask_dim, config.l_hidden_size / 2, batch_first=True, num_layers = config.l_num_layers / 2,
            bidirectional=True, dropout=config.l_dropout)
        init_ortho(self.rnn)

    # batch_size * sent_l * dim
    def forward(self, feats):
        #FIXIT: doesn't have batch
        feats = feats.unsqueeze(0)
        lstm_out, (hid_states, cell_states) = self.rnn(feats)

        #FIXIT: for batch
        lstm_out = lstm_out.squeeze(0)
        # batch * sent_l * 2 * hidden_states 
        return lstm_out

# input layer for 14
class SimpleCat(nn.Module):
    def __init__(self, config):
        super(SimpleCat, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(config.embed_num, config.embed_dim)
        self.mask_embed = nn.Embedding(2, config.mask_dim)

        self.dropout = nn.Dropout(config.dropout)

    # input are tensors
    def forward(self, sent, mask):
        sent = Variable(sent)
        mask = Variable(mask)
        if self.config.if_gpu:  sent, mask = sent.cuda(), mask.cuda()

        # to embeddings
        sent_vec = self.word_embed(sent) # sent_len * dim
        mask_vec = self.mask_embed(mask) # 1 * dim

        sent_vec = self.dropout(sent_vec)
        sent_vec = torch.cat([sent_vec, mask_vec], 1)

        # for test
        return sent_vec

    def load_vector(self):
        with open(self.config.embed_path) as f:
            vectors = cPickle.load(f)
            print "Loaded from {} with shape {}".format(self.config.embed_path, vectors.shape)
            self.word_embed.weight = nn.Parameter(torch.Tensor(vectors))
            # self.word_embed.weight.requires_grad = False
    
    def reset_binary(self):
        self.mask_embed.weight.data[0].zero_()
