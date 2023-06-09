# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         vqa_dataset
# Description:
# Author:       Boliu.Kelvin, Sedigheh Eslami
# Date:         2020/5/1
#-------------------------------------------------------------------------------
"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
from __future__ import print_function
import os
import json
import argparse
import _pickle as cPickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import itertools
import warnings
from PIL import Image
import h5py
import clip
import torch
import sys
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from utils.create_dictionary import Dictionary
from language.language_model import BERTWordEmbedding

import argparse
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
COUNTING_ONLY = False

img_name_key = 'image_name'


# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False



def _create_entry(img, data, answer):
    if None != answer:
        answer.pop(img_name_key)
        answer.pop('qid')
    entry = {
        'qid': data['qid'],
        'image_name': data['image_name'],
        'image': img,
        'question': data['question'],
        'answer': answer,
        'answer_text': data['answer'],
        'answer_type': data['answer_type'],
        'question_type': data['question_type'],
        'phrase_type': data['phrase_type']
    }
    return entry


def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError:
        return False
    return True


def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img2id: dict {img -> id} id can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    data_path = os.path.join(dataroot, name + 'set.json')
    samples = json.load(open(data_path))
    samples = sorted(samples, key=lambda x: int(x['qid']))

    answer_path = os.path.join(dataroot, 'cache', '%s_openclose_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: int(x['qid']))
    utils.assert_eq(len(samples), len(answers))
    entries = []
    for sample, answer in zip(samples, answers):
        utils.assert_eq(sample['qid'], answer['qid'])
        utils.assert_eq(sample['image_name'], answer[img_name_key])
        img_id = sample['image_name']
        if not COUNTING_ONLY or is_howmany(sample['question'], answer, label2ans):
            entries.append(_create_entry(img_id2val[img_id], sample, answer))

    return entries


class VQARADFeatureDataset(Dataset):

    def __init__(self, name, cfg, dictionary=None, dataroot='data', q_emb_model="bert-base-uncased"):
        super(VQARADFeatureDataset, self).__init__()

        # question
        self.q_emb_model = q_emb_model
        self.q_tokenizer = AutoTokenizer.from_pretrained(self.q_emb_model)

        question_len = cfg.TRAIN.QUESTION.LENGTH
        self.cfg = cfg
        self.name = name
        assert name in ['train', 'test']
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        # self.num_ans_candidates =  547 # 487 56 431

        # close & open
        self.label2close = cPickle.load(
            open(os.path.join(dataroot, 'cache', 'close_label2ans.pkl'), 'rb'))
        self.label2open = cPickle.load(
            open(os.path.join(dataroot, 'cache', 'open_label2ans.pkl'), 'rb'))
        self.num_open_candidates = len(self.label2open)
        self.num_close_candidates = len(self.label2close)

        self.num_ans_candidates = self.num_close_candidates + self.num_open_candidates
        # print(self.num_close_candidates, self.num_open_candidates, self.num_ans_candidates)

        # End get the number of answer type class
        self.dictionary = dictionary

        self.img_id2idx = json.load(open(os.path.join(dataroot, 'imgid2idx.json')))

        self.entries = _load_dataset(dataroot, name, self.img_id2idx, self.label2ans)

        # load image data for MAML module
        if self.cfg.TRAIN.VISION.MAML:

            images_path = os.path.join(dataroot, 'images84x84.pkl')
            print('loading MAML image data from file: ' + images_path)
            self.maml_images_data = cPickle.load(open(images_path, 'rb'))
        # load image data for Auto-encoder module
        if self.cfg.TRAIN.VISION.AUTOENCODER:

            images_path = os.path.join(dataroot, 'images128x128.pkl')
            print('loading DAE image data from file: ' + images_path)
            self.ae_images_data = cPickle.load(open(images_path, 'rb'))
        if self.cfg.TRAIN.VISION.CLIP:
            if self.cfg.TRAIN.VISION.CLIP_VISION_ENCODER == "RN50x4":
                images_path = os.path.join(dataroot, 'images288x288.pkl')
            else:
                print('loading 250x250')
                images_path = os.path.join(dataroot, 'images250x250.pkl')
            print(f"loading CLIP image data from file: {images_path}")
            self.clip_images_data = cPickle.load(open(images_path, 'rb'))

        # tokenization
        self.tokenize(question_len)
        self.tensorize()
        if cfg.TRAIN.VISION.AUTOENCODER and cfg.TRAIN.VISION.MAML:
            self.v_dim = cfg.TRAIN.VISION.V_DIM * 2
        else:
            self.v_dim = cfg.TRAIN.VISION.V_DIM  # see the V_DIM defined in config files

    def tokenize(self, max_length):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        TODO Revision: should be turned to 0 for padding_idx in embedding
        """

        for entry in self.entries:
            # tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = entry['question']
            tokens = tokens[:max_length]
            # if len(tokens) < max_length:
            #     padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
            #     tokens = tokens + padding

            # utils.assert_eq(len(tokens), max_length)
            # # assert no negative index enters the tokenized question
            # utils.assert_ge(min(tokens), 0)
            entry['q_token'] = tokens

        """


        bert <- input_ids, token_type_ids, attention_mask
                input_ids: (batch, seq_len)
                token_type_ids: (batch, seq_len)
                attention_mask: (batch, seq_len)

        collate_fn:
            batch = {
                # q needs to be separated into these 3 elements
                input_ids: (batch, seq_len),
                token_type_ids: (batch, seq_len),
                attention_mask: (batch, seq_len),
                v: (???),
                answer_type: (???),
                ... any v needed
            }

        bert_model(**batch)

        """
    # def bert_tokenize(self, max_length):
    #     """Tokenizes the questions.

    #     This will add q_token in each entry of the dataset.
    #     -1 represent nil, and should be treated as padding_idx in embedding
    #     TODO Revision: should be turned to 0 for padding_idx in embedding
    #     """

    #     for entry in self.entries:

    #         question = entry['question']
    #         encode = self.q_tokenizer(question,
    #                                   max_length=max_length,
    #                                   pad_to_max_length=True,
    #                                   truncation=True,
    #                                   )

    #         utils.assert_eq(len(encode['input_ids']), max_length)
    #         entry['q_token'] = encode

    def tensorize(self):
        if self.cfg.TRAIN.VISION.MAML:
            self.maml_images_data = torch.from_numpy(self.maml_images_data)
            self.maml_images_data = self.maml_images_data.type('torch.FloatTensor')
        if self.cfg.TRAIN.VISION.AUTOENCODER:
            self.ae_images_data = torch.from_numpy(self.ae_images_data)
            self.ae_images_data = self.ae_images_data.type('torch.FloatTensor')
        if self.cfg.TRAIN.VISION.CLIP:
            self.clip_images_data = torch.from_numpy(self.clip_images_data)
            self.clip_images_data = self.clip_images_data.type('torch.FloatTensor')
        for entry in self.entries:
            question = np.array(entry['q_token'])
            entry['q_token'] = question
            answer = entry['answer']
            if None != answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]

        answer = entry['answer']
        type = answer['type']
        answer_type = entry['answer_type']
        question_type = entry['question_type']
        phrase_type = entry['phrase_type']
        image_data = [0, 0, 0]
        if self.cfg.TRAIN.VISION.MAML:
            maml_images_data = self.maml_images_data[entry['image']].reshape(84 * 84)
            image_data[0] = maml_images_data
        if self.cfg.TRAIN.VISION.AUTOENCODER:
            ae_images_data = self.ae_images_data[entry['image']].reshape(128 * 128)
            image_data[1] = ae_images_data
        if self.cfg.TRAIN.VISION.CLIP:
            if self.cfg.TRAIN.VISION.CLIP_VISION_ENCODER == "RN50x4":
                clip_images_data = self.clip_images_data[entry['image']].reshape(3 * 288 * 288)
            else:
                clip_images_data = self.clip_images_data[entry['image']].reshape(3 * 250 * 250)
            image_data[2] = clip_images_data


        # question_data=[entry['q_token']]
        question_data = entry["question"] # str

        if answer_type == 'CLOSED':
            answer_target = 0
        else:
            answer_target = 1
        # print(self.num_ans_candidates, self.num_close_candidates,
        #   self.num_open_candidates)
        if None != answer:
            labels = answer['labels']
            scores = answer['scores']
            composed_target = torch.zeros(self.num_ans_candidates)  # close + open
            if answer_target == 0:
                target = torch.zeros(self.num_close_candidates)
                if labels is not None:
                    target.scatter_(0, labels, scores)
                composed_target[:self.num_close_candidates] = target
            else:
                target = torch.zeros(self.num_open_candidates)
                if labels is not None:
                    target.scatter_(0, labels, scores)
                composed_target[self.num_close_candidates:self.num_ans_candidates] = target
            if self.name == "test":
                return image_data, question_data, composed_target, answer_type, question_type, phrase_type, answer_target, entry[
                    'image_name'], entry['question'], entry['answer_text']
            else:
                return image_data, question_data, composed_target, answer_type, question_type, phrase_type, answer_target
        else:
            if self.name == "test":
                return image_data, question_data, answer_type, question_type, phrase_type, answer_target, entry[
                    'image_name'], entry['question'], entry['answer_text']
            else:
                return image_data, question_data, answer_type, question_type, phrase_type, answer_target

    def __len__(self):
        return len(self.entries)


    def collate_fn(batch):
        ...


def tfidf_from_questions(names, args, dictionary, dataroot='data', target=['rad']):
    inds = [[], []]  # rows, cols for uncoalesce sparse matrix
    df = dict()
    N = len(dictionary)
    if args.use_RAD:
        dataroot = args.RAD_dir

    def populate(inds, df, text):
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0])
                inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1])
                inds[1].append(c[0])

    if 'rad' in target:
        for name in names:
            assert name in ['train', 'test']
            question_path = os.path.join(dataroot, name + 'set.json')
            questions = json.load(open(question_path))
            for question in questions:
                populate(inds, df, question['question'])

    # TF-IDF
    vals = [1] * len(inds[1])
    for idx, col in enumerate(inds[1]):
        assert df[col] >= 1, 'document frequency should be greater than zero!'
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds, vals):
        z = dict()
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds), torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    # Latent word embeddings
    emb_dim = 300
    glove_file = os.path.join(dataroot, 'glove', 'glove.6B.%dd.txt' % emb_dim)
    weights, word2emb = utils.create_glove_embedding_init(dictionary.idx2word[N:], glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0), tfidf.size(1)))

    return tfidf, weights


if __name__ == '__main__':
    # dictionary = Dictionary.load_from_file('data_RAD/dictionary.pkl')
    # tfidf, weights = tfidf_from_questions(['train'], None, dictionary)
    # w_emb = WordEmbedding(dictionary.ntoken, 300, .0, 'c')
    # w_emb.init_embedding(os.path.join('data_RAD', 'glove6b_init_300d.npy'), tfidf, weights)
    # with open('data_RAD/embed_tfidf_weights.pkl', 'wb') as f:
    #     torch.save(w_emb, f)
    # print("Saving embedding with tfidf and weights successfully")

    # dictionary = Dictionary.load_from_file('data_RAD/dictionary.pkl')
    # w_emb = WordEmbedding(dictionary.ntoken, 300, .0, 'c')
    # with open('data_RAD/embed_tfidf_weights.pkl', 'rb') as f:
    #     w_emb = torch.load(f)
    # print("Load embedding with tfidf and weights successfully")
    #
    # # TODO: load img_id2idx
    # img_id2idx = json.load(open(os.path.join('./data_RAD', 'imgid2idx.json')))
    # label2ans_path = os.path.join('./data_RAD', 'cache', 'trainval_label2ans.pkl')
    # label2ans = cPickle.load(open(label2ans_path, 'rb'))
    # entries = _load_dataset('./data_RAD', 'train', img_id2idx, label2ans)
    # print(entries)

    from config import cfg, update_config
    default_cfg = 'configs/qcr_pubmedclipRN50_ae_rad_16batchsize_withtfidf_nondeterministic.yaml'
    parser = argparse.ArgumentParser(description="Med VQA")
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=default_cfg,
                        type=str)
    args = parser.parse_args()
    update_config(cfg, args)
    print(cfg)
    dataroot = './data_rad'

    # d = Dictionary.load_from_file(os.path.join(dataroot,'dictionary.pkl'))
    print("Loading BERT dictionary")
    d = Dictionary.load_from_model_name(model_name='bert-base-uncased')
    dataset = VQARADFeatureDataset('train', cfg, d, dataroot)
    train_data = DataLoader(dataset,
                            batch_size=20,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True,
                            drop_last=False)
    for i, row in enumerate(train_data):
        # image_data, question_data, answer_type, question_type, phrase_type, answer_target, entry['image_name'], entry['question'], entry['answer_text']
        print(row)
