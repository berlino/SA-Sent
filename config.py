from __future__ import division

class Config():
    def __init__(self):
        # for reader
        self.batch_size = 10

        self.embed_num = 5135
        self.embed_dim = 300
        self.mask_dim = 50
        self.if_update_embed = False

        # lstm
        self.l_hidden_size = 200
        self.l_num_layers = 2 # forward and backward
        self.l_dropout = 0.1

        # penlaty
        self.C1 = 0.1
        self.C2 = 0
        self.if_reset = True

        self.opt = "Adam"
        self.dropout = 0.4
        self.epoch = 20
        self.lr = 0.01 / self.batch_size
        self.l2 = 0.0
        self.adjust_every = 10
        self.clip_norm = 3
        self.k_fold = 6

        # data processing
        self.if_replace = False

        # traning
        self.if_gpu = False

        # self.embed_path = "./data/2016/pre-trained-google.pkl"
        self.embed_path = "./data/2014/pre-trained-glove.pkl"
        self.data_path = "./data/2014/data.pkl"
        self.dic_path = "./data/2014/dic.pkl"

    
    def __repr__(self):
        return str(vars(self))

config = Config()