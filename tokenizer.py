#!/usr/bin/python
from stanford_corenlp_pywrapper import CoreNLP
import pdb

proc = CoreNLP("ssplit", corenlp_jars=["/Users/wangbolin/Desktop/DATASET/stanford-corenlp-python/stanford-corenlp-full-2014-08-27/*"])

def tokenize(string):
    parse_ret = proc.parse_doc(string)
    ret_l = []
    sents = parse_ret["sentences"]
    for sent in sents:
        ret_l.extend(sent["tokens"])
    return ret_l

if __name__ == "__main__":
    tokenize("hello world!")