from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import util 
import numpy as np
import time

class LinearCRF(nn.Module):
    def __init__(self, config):
       super(LinearCRF, self).__init__()
       self.config = config
       self.label_size = 2

       #T[i,j] for j to i, not i to j
       self.transitions = nn.Parameter(torch.randn(self.label_size, self.label_size))
    
    # no batch size
    def predict(self, feats):
        feats = feats.unsqueeze(0)
        return self._viterbi_decode(feats)[0]

    # feats: batch * sent_l * label_size
    # labels: batch * sent_l
    def _score_sentence(self, feats, labels):
        batch_size, sent_l, label_size = feats.size()

        # sent_l * label_size * batch_size
        feats = feats.transpose(0,1).transpose(1,2).contiguous()

        # TODO advanced index?
        scores = []
        for batch_id, inst in enumerate(labels):
            score = util.create_empty_var(feats.is_cuda)
            for i in range(sent_l):
                if i == 0:
                    score += feats[i, inst[i], batch_id]
                else:
                    score += feats[i, inst[i], batch_id] + self.transitions[inst[i], inst[i - 1]]
            scores.append(score)
        return torch.cat(scores)
    
    # feats: batch * sent_l * feats
    def _forward_alg(self, feats):
        batch_size, sent_len, _ = feats.size()
        # the first row should always be zero
        init_alphas = torch.Tensor(sent_len + 1, batch_size, self.label_size).fill_(0)
        if feats.is_cuda:  init_alphas = init_alphas.cuda()
        # forward_var[i][j] means message ends at token i(excluded) with label j
        forward_var = Variable(init_alphas)
        # for the convenience of index   #sent_l * feats * batch
        feats = feats.transpose(0,1).transpose(1,2).contiguous()

        # points to i+1 node
        for i in range(sent_len):
            #if i == 4: pdb.set_trace()
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.label_size):
                next_tag_var = None
                feat = feats[i, next_tag]
                # batch_size * label_size
                if i == 0: # the first and last node don't have the transition score
                    messa = feat.view(batch_size, 1)
                    alphas_t.append(messa)
                else:
                    emit_score = feat.view(batch_size, 1).expand(batch_size, self.label_size)
                    trans_score = self.transitions[next_tag].view(1, self.label_size).expand(batch_size, self.label_size)
                    next_tag_var = forward_var[i] + trans_score + emit_score
                    messa = util.log_sum_exp_m(next_tag_var) 
                    messa = messa.view(batch_size, 1)
                    alphas_t.append(messa)
            forward_var[i + 1] = torch.cat(alphas_t, 1).view(batch_size, self.label_size)
        terminal_var = forward_var[-1]
        # pdb.set_trace()
        # batch_size 1d tensor
        alpha = util.log_sum_exp_m(terminal_var)
        return alpha, forward_var[1:].squeeze(1)

    # for sanity check
    def _backward_alg(self, feats):
        batch_size, sent_len, _ = feats.size()
        # the last row should always be zero
        init_betas = torch.Tensor(sent_len + 1, batch_size, self.label_size).fill_(0)
        if feats.is_cuda:  init_betas = init_betas.cuda()
        # backward_var[i][j] means message starts from token i(included) with label j
        backward_var = Variable(init_betas)

        feats = feats.transpose(0,1).transpose(1,2).contiguous()

        # pdb.set_trace()
        for i in reversed(range(sent_len)):
            betas_t = []  # The forward variables at this timestep
            for pre_tag in range(self.label_size):
                #pre_tag_var = Variable(torch.Tensor(1, self.label_size).fill_(0))
                pre_tag_var = None
                feat = feats[i, pre_tag]
                trans_score = self.transitions.transpose(0,1).contiguous()[pre_tag].view(1, self.label_size).expand(batch_size, self.label_size)
                if i + 1 == sent_len:
                    messa = feat.view(batch_size, 1)
                    betas_t.append(messa)
                else:
                    emit_score = feat.view(batch_size, 1).expand(batch_size, self.label_size)
                    pre_tag_var = backward_var[i+1] + trans_score + emit_score
                    messa = util.log_sum_exp_m(pre_tag_var)
                    messa = messa.view(batch_size, 1)
                    betas_t.append(messa)
            backward_var[i] = torch.cat(betas_t, 1).view(batch_size, self.label_size)
        terminal_var = backward_var[0]
        #pdb.set_trace()
        beta = util.log_sum_exp_m(terminal_var)
        return beta, backward_var[:-1].squeeze(1)
    
    # feats is batch * sent_l * label_size
    def _viterbi_decode(self, feats):
        batch_size, sent_len, _ = feats.size()        
        feats = feats.transpose(0,1).transpose(1,2).contiguous()

        # it should finally with the size: sent_len * label_size * batch_size
        pointers = []
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(sent_len + 1, batch_size, self.label_size).fill_(0)
        if feats.is_cuda:  init_vvars = init_vvars.cuda() 
        forward_var = Variable(init_vvars)

        # pdb.set_trace()
        pointers = []
        for i in range(sent_len):
            # label_size * batch_size
            viterbivars_t = []
            bptr_s = []

            for next_tag in range(self.label_size):
                #next_tag_var = Variable(torch.Tensor(1, self.label_size).fill_(0))
                next_tag_var = None
                feat = feats[i, next_tag]
                emit_score = feat.view(batch_size, 1).expand(batch_size, self.label_size)
                if i == 0: # the first node don't have the transition score
                    next_tag_var = forward_var[i] + emit_score
                else:
                    trans_score = self.transitions[next_tag].view(1, self.label_size).expand(batch_size, self.label_size)
                    next_tag_var = forward_var[i] + trans_score + emit_score
                # pdb.set_trace()
                best_ids, best_value = util.argmax_m(next_tag_var)
                bptr_s.append(best_ids)
                best_value = best_value.view(-1, 1)
                viterbivars_t.append(best_value)
            forward_var[i + 1] = torch.cat(viterbivars_t, 1).view(batch_size, self.label_size)
            pointers.append(bptr_s)
    

        # pdb.set_trace()
        # decode the pointers
        assert len(pointers) == sent_len
        assert len(pointers[0]) == self.label_size
        # should be batch_size * sent_len
        pointers = np.array(pointers)
        ret_label = []
        for batch_id in range(batch_size):
            final_label = util.argmax(forward_var[-1, batch_id])
            sent_labels = []
            # the first state should always be zero
            seqs = pointers[1:,:, batch_id]
            f_ = final_label
            sent_labels.append(f_)
            for tok_ind in reversed(range(sent_len - 1)):
                f_ = seqs[tok_ind][f_]
                sent_labels.append(f_)
            # remember to reverse the labels
            ret_label.append(list(reversed(sent_labels)))
        
        return ret_label

    def neg_log_likelihood(self, feats, labels):
        forward_score = self._forward_alg(feats)
        # backward_score = self._backward_alg(feats)
        gold_score = self._score_sentence(feats, labels)
        loss_vec = forward_score - gold_score
        # using average
        return loss_vec.mean()

    def reset_transition(self):
        # self.transitions.data[0,0] = 0.5
        # self.transitions.data[1,1] = 1
        # self.transitions.data[0,1] = -0.5
        # self.transitions.data[1,0] = -0.5
        pass


    def forward(self, feats):
        sent_len, feat_dim = feats.size()
        i_feats = feats.unsqueeze(0)
        Z1, forward_mat = self._forward_alg(i_feats) 
        Z2, backward_mat = self._backward_alg(i_feats)
        # assert Z1[0] == Z2[0]
        forward_v = forward_mat
        backward_v = backward_mat

        message_v = forward_v + backward_v - feats
        Z = Z1.expand(sent_len * self.label_size).contiguous().view(sent_len, self.label_size)
        marginal_v = torch.exp(message_v - Z)

        return marginal_v

        
