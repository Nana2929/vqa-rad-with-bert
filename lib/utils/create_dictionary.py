# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         create_dictionary
# Description:  question->word->dictionary for validation & training
# Author:       Boliu.Kelvin
# Date:         2020/4/5
#-------------------------------------------------------------------------------

import argparse
import json
import sys
import os
import pandas as pd
import numpy as np
import _pickle as cPickle
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Dictionary(object):
    """Revised by Ching Wen Yang, 2023/06/02
    """

    def __init__(self, word2idx=None, idx2word=None, bert_model_name: str = None):


        # load tokenizer
        if bert_model_name is not None:
            self.bert_model_name = bert_model_name
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            # growing dictionary
            self.word2idx = self.tokenizer.vocab
            self.idx2word = list(self.tokenizer.vocab.keys())
            self.padding_idx = self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]

        elif word2idx and idx2word:
            self.word2idx = word2idx
            self.idx2word = idx2word
            self.padding_idx = len(self.idx2word)

    @property
    def ntoken(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        if "? -yes/no" in sentence:
            sentence = sentence.replace("? -yes/no", "")
        if "? -open" in sentence:
            sentence = sentence.replace("? -open", "")
        if "? - open" in sentence:
            sentence = sentence.replace("? - open", "")
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace(
            '...', '').replace('x ray', 'x-ray').replace('.', '')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # if a word is not in dictionary, it will be replaced with the last word of dictionary.
                tokens.append(self.word2idx.get(w, len(self.word2idx) - 1))

        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print(f'tokenizer dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading tokenizer dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx=word2idx, idx2word=idx2word)
        return d

    @classmethod
    def load_from_model_name(cls, model_name: str):
        print('loading BERT tokenizer dictionary from %s' % model_name)
        d = cls(bert_model_name=model_name)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def create_bert_dictionary(bert_model_name: str):
    return Dictionary.load_from_model_name(bert_model_name)



def create_dictionary(dataroot, dataset_name, train_file, test_file):
    dictionary = Dictionary()
    questions = []
    files = [train_file, test_file]
    for path in files:
        data_path = os.path.join(data, path)
        with open(data_path) as f:
            d = json.load(f)
        df = pd.DataFrame(d)
        if dataset_name.lower() in ["slake", "vqa-slake", "vqa_slake"]:
            df = df[df['q_lang']=="en"]
        print("processing the {}".format(path))
        for id, row in df.iterrows():
            dictionary.tokenize(row['question'], True)     #row[0]: id , row[1]: question , row[2]: answer

    return dictionary

def create_glove_embedding_init(idx2word, glove_file):
    print('creating glove embeddings...')
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Med VQA")
    parser.add_argument("--inputpath", type=str, help="./data_rad")
    parser.add_argument("--dataset", type=str, help="Name of the dataset", default="rad")
    parser.add_argument("--trainfile", type=str, help="Name of the train file", default="train.json")
    parser.add_argument("--testfile", type=str, help="Name of the test file", default="test.json")
    args = parser.parse_args()
    data = args.inputpath
    dataset = args.dataset
    train_file = args.trainfile
    test_file = args.testfile
    d = create_bert_dictionary('bert-base-uncased')
    d.dump_to_file(data + '/dictionary.pkl')

    d = Dictionary.load_from_file(data + '/dictionary.pkl')
    print("dictionary size: {}".format(len(d)))
    # emb_dim = 300

    # glove_file = f'/home/nanaeilish/glove/glove.6B.{emb_dim}d.txt'
    # weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    # np.save(data + '/glove6b_init_%dd.npy' % emb_dim, weights)

