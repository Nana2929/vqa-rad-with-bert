# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         language_model
# Description:  This code is from Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang's repository.
# https://github.com/jnhwkim/ban-vqa
# Author:       Boliu.Kelvin
# Date:         2020/4/7
#-------------------------------------------------------------------------------

from typing import List
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig


class BERTWordEmbedding(nn.Module):
    """BERT Word Embedding
    """

    def __init__(self, bert_model_name, dropout=0.5, max_length=20):
        super(BERTWordEmbedding, self).__init__()

        self.tokenizer =  AutoTokenizer.from_pretrained(bert_model_name)
        self.config = AutoConfig.from_pretrained(bert_model_name)
        self.model = AutoModel.from_pretrained(bert_model_name, config=self.config)

        self.max_length = max_length
        self.output_layer = nn.Linear(self.config.hidden_size, 1024)
        self.vocab_size = self.config.vocab_size
        self.emb_dim = self.config.hidden_size
        # no longer used but the pretrained checkpoint requires this
        self.emb = nn.Embedding(self.vocab_size + 1,
                                self.emb_dim,
                                padding_idx=self.find_padding_idx())
        self.emb.weight.requires_grad = True # fixed
       
    def find_padding_idx(self):
        pad_id = self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        return pad_id

    # Archive
    def _prepare_last_4(self, x: str):
        """
        x: a string/question
        """

        tokenized_text = self.tokenizer.tokenize(x)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)


        # Stores the token vectors, with shape [22 x 3,072]
        token_vecs_sum = []
        # `token_embeddings` is a [12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)
            # Use `cat_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)
        return token_vecs_sum  # shape [seq_len, 768]

    def is_bert_training(self):
        return self.model.training

    def fully_forward(self, input_ids, token_type_ids=None, attention_mask=None, **kwargs):
        # (batch_size, seq_len, 1024)
        return self.output_layer(self.model(input_ids, token_type_ids, attention_mask))

    def encode_and_cast_dim_forward(self, questions:List[str]):
        # (batch_size, seq_len, 1024)
        # 1024: HID_DIM
        return  self.output_layer(self.encode_and_forward(questions)) # (batch_size, seq_len, 1024)  as type attention

    # def type_attn_forward(self, questions:List[str]):
    #     # for TypeAttention
    #     # (batch_size, 1024)
    #     # 1024: HID_DIM
    #     return torch.sum(self.encode_and_cast_dim_forward(questions), dim=1).squeeze()


    def emb_forward(self, questions: List[str]):
        # use first layer output as embedding
        device = next(self.model.parameters()).device
        tokenized = self.tokenizer.batch_encode_plus(questions,
                                                  return_tensors="pt",
                                                  max_length=self.max_length,
                                                  truncation=True,
                                                  padding='max_length')
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        # use BERT embedding layer (0th layer) output
        embeddings = self.model.embeddings(tokenized['input_ids'],
                                           tokenized['attention_mask'])
        # (batch_size, seq_len, 768)

        return embeddings

    def encode_and_forward(self, questions:List[str]):
        # use last layer output as embedding
        # (batch_size, seq_len, 768)

        device = next(self.model.parameters()).device
        encode = self.tokenizer.batch_encode_plus(questions,
                                                  return_tensors="pt",
                                                  max_length=self.max_length,
                                                  truncation=True,
                                                  padding='max_length'
                                                  )

        encode = {k: v.to(device) for k, v in encode.items()}
        # https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
        logit = self.model(**encode).last_hidden_state # (batch_size, seq_len, 768)
        return logit

class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """

    def __init__(self, ntoken, emb_dim, dropout, cat=True):
        super(WordEmbedding, self).__init__()
        self.cat = cat
        self.emb = nn.Embedding(ntoken + 1, emb_dim, padding_idx=ntoken)
        if cat:
            self.emb_ = nn.Embedding(ntoken + 1, emb_dim, padding_idx=ntoken)
            self.emb_.weight.requires_grad = False  # fixed
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file, tfidf=None, tfidf_weights=None):
        weight_init = torch.from_numpy(np.load(np_file))
        print(weight_init.shape)
        # print(self.ntoken, self.emb_dim)
        # assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init
        if tfidf is not None:
            if 0 < tfidf_weights.size:
                weight_init = torch.cat([weight_init, torch.from_numpy(tfidf_weights)], 0)
            weight_init = tfidf.matmul(weight_init)  # (N x N') x (N', F)
            self.emb_.weight.requires_grad = True
        if self.cat:
            self.emb_.weight.data[:self.ntoken] = weight_init.clone()

    def forward(self, x):
        emb = self.emb(x)
        if self.cat:
            emb = torch.cat((emb, self.emb_(x)), 2)
        emb = self.dropout(emb)  # output shape: (batch_size, seq_len, emb_dim)
        return emb


class QuestionEmbedding(nn.Module):

    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU if rnn_type == 'GRU' else None

        self.rnn = rnn_cls(in_dim,
                           num_hid,
                           nlayers,
                           bidirectional=bidirect,
                           dropout=dropout,
                           batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid // self.ndirections)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:  # GRU
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        # hidden(GRU): [1, batch, num_hid]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        # For unbatched 2-D input, hx should also be 2-D but got 3-D tensor

        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        output, hidden = self.rnn(x, hidden)
        return output
