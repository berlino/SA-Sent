from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb
import cPickle
from CRF import LinearCRF
import math
from util import *
from Layer import MLSTM, SimpleCat

# consits of three components
class AspectSent(nn.Module):
    def __init__(self, config):
        super(AspectSent, self).__init__()
        self.config = config
        self.cat_layer = SimpleCat(config)

        self.lstm = MLSTM(config)
        self.feat2tri = nn.Linear(config.l_hidden_size, 2)
        self.inter_crf = LinearCRF(config)
        self.feat2label = nn.Linear(config.l_hidden_size, 3)

        self.cri = nn.CrossEntropyLoss()
        self.cat_layer.load_vector()

        if not config.if_update_embed:  self.cat_layer.word_embed.weight.requires_grad = False

    
    def compute_scores(self, sent, mask):
        if self.config.if_reset:  self.cat_layer.reset_binary()
        # self.inter_crf.reset_transition()

        sent = torch.LongTensor(sent)
        mask = torch.LongTensor(mask)
        sent_vec = self.cat_layer(sent, mask)

        context = self.lstm(sent_vec)

        # feat_context = torch.cat([context, asp_v], 1) # sent_len * dim_sum
        feat_context = context  # sent_len * dim_sum
        tri_scores = self.feat2tri(feat_context)
        marginals = self.inter_crf(tri_scores)
        select_polarity = marginals[:,1]

        marginals = marginals.transpose(0,1)  # 2 * sent_len
        sent_v = torch.mm(select_polarity.unsqueeze(0), context) # 1 * feat_dim
        label_scores = self.feat2label(sent_v).squeeze(0)

        return label_scores, select_polarity, marginals

    def compute_predict_scores(self, sent, mask):
        if self.config.if_reset:  self.cat_layer.reset_binary()
        # self.inter_crf.reset_transition()

        sent = torch.LongTensor(sent)
        mask = torch.LongTensor(mask)
        sent_vec = self.cat_layer(sent, mask)

        context = self.lstm(sent_vec)

        # feat_context = torch.cat([context, asp_v], 1) # sent_len * dim_sum
        feat_context = context  # sent_len * dim_sum
        tri_scores = self.feat2tri(feat_context)
        marginals = self.inter_crf(tri_scores)
        select_polarity = marginals[:,1]

        best_seqs = self.inter_crf.predict(tri_scores)

        sent_v = torch.mm(select_polarity.unsqueeze(0), context) # 1 * feat_dim
        label_scores = self.feat2label(sent_v).squeeze(0)

        return label_scores, select_polarity, best_seqs

    
    def forward(self, sent, mask, label):
        '''
        inputs are list of list for the convenince of top CRF
        '''
        # scores = self.compute_scores(sents, ents, asps, labels)
        scores, s_prob, marginal_prob = self.compute_scores(sent, mask)

        pena = F.relu( self.inter_crf.transitions[1,0] - self.inter_crf.transitions[0,0]) + \
            F.relu(self.inter_crf.transitions[0,1] - self.inter_crf.transitions[1,1])
        norm_pen = ( self.config.C1 * pena + self.config.C2 * s_prob.norm(1) ) / self.config.batch_size

        scores = F.softmax(scores)
        cls_loss = -1 * torch.log(scores[label])

        print "cls loss {0} with penalty {1}".format(cls_loss.data[0], norm_pen.data[0])
        return cls_loss + norm_pen 

    def predict(self, sent, mask):
        scores, s_probs, best_seqs = self.compute_predict_scores(sent, mask)
        _, pred_label = scores.max(0)        

        return pred_label.data[0], best_seqs