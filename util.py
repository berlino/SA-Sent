from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # vec is only 1d vec
    # return the argmax as a python int
    _, idx = torch.max(vec, 0)
    return to_scalar(idx)

# the input is 2d dim tensor
# output 1d tensor
def argmax_m(mat):
    ret_v, ret_ind = [], []
    m, n = mat.size()
    for i in range(m):
        ## cj modified
        ind_ = argmax(mat[i])
        ret_ind.append(ind_)
        ret_v.append(torch.tensor([mat[i][ind_]]))
    if type(ret_v[0]) == Variable or type(ret_v[0]) == torch.Tensor:
        return ret_ind, torch.cat(ret_v)
    else:
        return ret_ind, torch.Tensor(ret_v)

# Compute log sum exp in a numerically stable way for the forward algorithm
# vec is n * n, norm in row

def log_sum_exp_m(mat):
    row, column = mat.size()
    ret_l = []
    for i in range(row):
        vec = mat[i]
        max_score = vec[argmax(vec)]
        max_score_broadcast = max_score.view( -1).expand(1, vec.size()[0])

        ## cj modified
        v = max_score + \
                     torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
        ret_l.append(torch.tensor([v]))

    return torch.cat(ret_l, 0)

def log_sum_exp(vec_list):
    tmp_mat = torch.stack(vec_list, 0)
    m,n = tmp_mat.size()
    # value may be nan because of gradient explosion
    try:
        max_score = torch.max(tmp_mat)
    except:
        pdb.set_trace()
    max_expand = max_score.expand(m, n)
    max_ex_v = max_score.expand(1, n)
    # sum along dim 0
    ret_val = max_ex_v + torch.log(torch.sum(torch.exp(tmp_mat - max_expand), 0))
    return ret_val




# vec1 and vec2 both 1d tensor
# return 1d tensor
def add_broad(vec1, vec2):
    s_ = vec1.size()[0]
    vec1 = vec1.expand(3, s_).transpose(0,1)
    vec2 = vec2.expand(s_, 3)
    new_vec = vec1 + vec2
    return new_vec.view(-1)

# transform a list to 1d vec
def to_1d(vec_list):
    ret_v = vec_list[0].clone()
    v_l = len(vec_list)
    for i in range(1, v_l):
        ret_v = add_broad(ret_v, vec_list[i])
    return ret_v

def to_ind(num, logit):
    ret_l = []
    for i in reversed(range(logit)):
        tmp = num / 3 ** i
        num = num - tmp * 3 ** i
        ret_l.append(tmp)
    return list(reversed(ret_l))

def create_empty_var(if_gpu):
    if if_gpu:
        loss = Variable(torch.Tensor([0]).cuda())
    else:
        loss = Variable(torch.Tensor([0])) 
    return loss
