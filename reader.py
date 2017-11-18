#!/usr/bin/python
from collections import namedtuple, defaultdict
import codecs
from config import config
from bs4 import BeautifulSoup
import pdb
import torch
from nltk.tokenize import word_tokenize
import tokenizer
import numpy as np
import re
import cPickle
import random

SentInst = namedtuple("SentenceInstance", "id text text_inds opinions")
OpinionInst = namedtuple("OpinionInstance", "target_text polarity class_ind target_mask")

TRAIN_DATA_PATH = "./data/2014/Restaurants_Train_v2.xml"
TEST_DATA_PATH = "./data/2014/Restaurants_Test_Gold.xml"

#TRAIN_DATA_PATH = "./data/2014/Laptop_Train_v2.xml"
#TEST_DATA_PATH = "./data/2014/Laptops_Test_Gold.xml"

GLOVE_FILE = "./data/glove/glove.840B.300d.txt"
OUT_FILE = config.embed_path
DATA_FILE = config.data_path
DIC_FILE = config.dic_path

class Reader():
    def __init__(self, config):
        self.config = config

        # id map to instance
        self.id2word = []
        self.word2id = {}
        self.id2label = ["positive", "neutral", "negative"]
        self.label2id = {v:k for k,v in enumerate(self.id2label)}

        self.UNK = "UNK"
        self.EOS = "EOS"

        # data
        self.train_data = None
        self.test_data = None
    
    def read_data(self, file_name):
        f = codecs.open(file_name, "r", encoding="utf-8")
        soup = BeautifulSoup(f.read(), "lxml")
        sentence_tags = soup.find_all("sentence")

        sentence_list = []
        for sent_tag in sentence_tags:
            sent_id = sent_tag.attrs["id"]
            sent_text = sent_tag.find("text").contents[0]
            opinion_list = []
            try:
                asp_tag = sent_tag.find_all("aspectterms")[0]
            except:
                # print "{0} {1} has no opinions".format(sent_id, sent_text)
                continue
            opinion_tags = asp_tag.find_all("aspectterm")
            for opinion_tag in opinion_tags:
                term = opinion_tag.attrs["term"]
                if term not in sent_text: pdb.set_trace()
                polarity = opinion_tag.attrs["polarity"]
                opinion_inst = OpinionInst(term, polarity, None, None)
                opinion_list.append(opinion_inst)
            sent_Inst = SentInst(sent_id, sent_text, None, opinion_list)
            sentence_list.append(sent_Inst)

        return sentence_list

    # generate vocabulary
    def gen_dic(self):
        words_set = set()
        label_set = set()

        # unknow
        words_set.add(self.UNK)

        for data in [self.train_data, self.test_data]:
            sent_counter = 0
            for sent_inst in data:
                sent_counter += 1
                tokens = self.tokenize(sent_inst.text)
                # pdb.set_trace()
                for token in tokens:
                    if token not in words_set:
                        words_set.add(token)
            print "{0} sentences".format(sent_counter)
        self.id2word = list(words_set)
        self.word2id = {v:k for k,v in enumerate(self.id2word)}

        print "{0} tokens".format(self.id2word.__len__())

    def tokenize(self, sent_str):
        # return word_tokenize(sent_str)
        sent_str = " ".join(sent_str.split("-"))
        sent_str = " ".join(sent_str.split("/"))
        sent_str = " ".join(sent_str.split("!"))
        return tokenizer.tokenize(sent_str)
        
    # namedtuple is protected!
    def to_index(self, data):
        sent_len = len(data)
        for sent_i in range(sent_len):
            sent_inst = data[sent_i]
            sent_tokens = self.tokenize(sent_inst.text)
            sent_inds = [self.word2id[x] if x in self.word2id else self.word2id[self.UNK] 
                for x in sent_tokens]
            if sent_inds is None: pdb.set_trace()
            sent_inst = sent_inst._replace(text_inds = sent_inds)

            opinion_list = []
            opi_len = len(sent_inst.opinions)
            for opi_i in range(opi_len):
                opi_inst = sent_inst.opinions[opi_i]

                target = opi_inst.target_text
                target_tokens = self.tokenize(target)
                try:
                    target_start = sent_tokens.index(target_tokens[0])
                    target_end = sent_tokens[max(0, target_start - 1):].index(target_tokens[-1])  + max(0, target_start - 1)
                except:
                    pdb.set_trace()
                if target_start < 0 or target_end < 0:  pdb.set_trace()
                mask = [0] * len(sent_tokens)
                for m_i in range(target_start, target_end + 1):
                    mask[m_i] = 1

                label = opi_inst.polarity
                if label == "conflict":  continue  # ignore conflict ones
                opi_inst = opi_inst._replace(class_ind = self.label2id[label])
                opi_inst = opi_inst._replace(target_mask = mask)
                opinion_list.append(opi_inst)
            
            sent_inst = sent_inst._replace(opinions = opinion_list)
            
            data[sent_i] = sent_inst

    
    def read(self):
        self.train_data = self.read_data(TRAIN_DATA_PATH)
        self.test_data = self.read_data(TEST_DATA_PATH)
        self.gen_dic()
        self.to_index(self.train_data)
        self.to_index(self.test_data)
        return self.train_data, self.test_data

    # shuffle and to batch size
    def to_batches(self, data, if_batch = False):
        all_triples = []
        # list of list
        pair_couter = defaultdict(int)
        print "Sentence size ", len(data)
        for sent_inst in data:
            tokens = sent_inst.text_inds
            
            for opi_inst in sent_inst.opinions:
                if opi_inst.polarity is None:  continue # conflict one
                mask = opi_inst.target_mask
                polarity = opi_inst.class_ind
                if tokens is None or mask is None or polarity is None: pdb.set_trace()
                all_triples.append([tokens, mask, polarity])
                pair_couter[polarity] += 1
        print pair_couter

        if if_batch:
            random.shuffle(all_triples)
            batch_n = len(all_triples) / self.config.batch_size + 1
            print "{0} instances with {1} batches".format(len(all_triples), batch_n)
            ret_triples = []
            
            offset = 0
            for i in range(batch_n):
                start = self.config.batch_size * i
                end = min(self.config.batch_size * (i+1), len(all_triples) )
                ret_triples.append(all_triples[start : end])
            return ret_triples
        else:
            return all_triples

    def gen_vectors_glove(self):
        vocab_dic = {}
        with open(GLOVE_FILE) as f:
            for line in f:
                s_s = line.split()
                if s_s[0] in self.word2id:
                    vocab_dic[s_s[0]] = np.array([float(x) for x in s_s[1:]])

        unknowns = np.random.uniform(-0.01, 0.01, 300).astype("float32")
        ret_mat = []
        unk_counter = 0
        for token in self.id2word:
            # token = token.lower()
            if token in vocab_dic:
                ret_mat.append(vocab_dic[token])
            else:
                ret_mat.append(unknowns)
                # print token
                unk_counter += 1
        ret_mat = np.array(ret_mat)
        with open(OUT_FILE, "w") as f:
            cPickle.dump(ret_mat, f)
        print "{0} unk out of {1} vocab".format(unk_counter, len(self.id2word))        
    
    def load_vectors(self):
        with open(OUT_FILE) as f:
            self.id2vec = cPickle.load(f)
    
    def debug_single_sample(self, batches, batch_n, sent_n):
        sent_ind = batches[batch_n][sent_n][0]
        print " ".join([self.id2word[x] for x in sent_ind])
        mask = batches[batch_n][sent_n][1]
        print mask
        label = batches[batch_n][sent_n][2]
        print self.id2label[label]
        
    
if __name__ == "__main__":
    reader = Reader(config)
    train, test = reader.read()

    train_batch = reader.to_batches(train, True)
    test_batch = reader.to_batches(test)

    reader.debug_single_sample(train_batch, 0, 0)
    # pdb.set_trace()

    # write to pkl
    with open(DATA_FILE, "w") as f:
        cPickle.dump([train_batch, test_batch],f)
    
    with codecs.open(DIC_FILE, 'w', encoding="utf-8") as f:
        cPickle.dump(reader.id2word, f)
        
    reader.gen_vectors_glove()


        

        
