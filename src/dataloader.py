# Load Library
import math
import os
from collections import Counter
import random

import numpy as np
import torch

class NERSequence(torch.utils.data.Dataset):

    def __init__(self, x, y, batch_size=1, preprocess=None, shuffle=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.preprocess = preprocess
        
        if shuffle:
            x_y = list(zip(self.x, self.y))
            random.shuffle(x_y)
            self.x, self.y = zip(*x_y)
        
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
        
        return self.preprocess(batch_x, batch_y)
    
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


class DataLoader:
    def __init__(self, anno_format):
        assert anno_format in ['conll', 'brat', 'pubchem'], "The Annotation Format Does Not Correspond This Format"
        self.format = anno_format #{conll, brat, pubchem}

    def load_data(self, path, encode='utf-8'):
        assert path is not None, "The Data Path Is Not Allow Empty"
        if self.format == 'conll':
            sents, labels = self._load_conll(path, encode)
        elif self.format == 'brat':
            sents, labels = self._load_brat(path, encode)
        else: # <- 'pubchem'
            sents, labels = self._load_pubchem(path, encode)
            
        return sents, labels
    
    def _load_conll(self, path, encode):
        sents, labels = [], []
        words, tags = [], []
        with open(path, encoding=encode) as f:
            for line in f:
                line = line.rstrip()
                if line:
                    word, tag = line.split(' ')
                    words.append(word)
                    tags.append(tag)
                else:
                    sents.append(words)
                    labels.append(tags)
                    words, tags = [], []
                    
        return sents, labels
        
    def _load_brat(self, path, encode):
        sents, labels = [], []
        words, tags = [], []
        #with open(self.path, encoding=self.encode) as f:
            
        return sents, labels
        
    def _load_pubchem(self, path, encode):
        sents, labels = [], []
        words, tags = [], []
        #with open(self.path, encoding=self.encode) as f:
        
        return sents, labels


class Vocabulary(object):
    def __init__(self, max_size=None, lower=True, unk_token=True, specials=('<pad>',)):
        self._max_size = max_size
        self._lower = lower
        self._unk = unk_token
        self._token2id = {token: i for i, token in enumerate(specials)}
        self._id2token = list(specials)
        self._token_count = Counter()
        
    def __len__(self):
        return len(self._token2id)
    
    def add_token(self, token):
        token = self.process_token(token)
        self._token_count.update([token])
        
    def add_documents(self, docs):
        for sent in docs:
            sent = map(self.process_token, sent)
            self._token_count.update(sent)
            
    def doc2id(self, doc):
        doc = map(self.process_token, doc)
        return [self.token_to_id(token) for token in doc]

    def id2doc(self, ids):
        return [self.id_to_token(idx) for idx in ids]
    
    def build(self):
        token_freq = self._token_count.most_common(self._max_size)
        idx = len(self.vocab)
        for token, _ in token_freq:
            self._token2id[token] = idx
            self._id2token.append(token)
            idx += 1
        if self._unk:
            unk = '<unk>'
            self._token2id[unk] = idx
            self._id2token.append(unk)
            
    def process_token(self, token):
        if self._lower:
            token = token.lower()
            
        return token
    
    def token_to_id(self, token):
        token = self.process_token(token)
        return self._token2id.get(token, len(self._token2id) -1)
    
    def id_to_token(self, idx):
        return self._id2token[idx]
    
    @property
    def vocab(self):
        return self._token2id
    
    def reverse_vocab(self):
        return self._id2token

	
def filter_embeddings(embeddings, vocab, dim):
    if not isinstance(embeddings, dict):
        return
    _embeddings = np.zeros([len(vocab), dim])
    for word in vocab:
        if word in embeddings:
            word_idx = vocab[word]
            _embeddings[word_idx] = embeddings[word_idx]
                
    return _embeddings


def load_glove(file):
    with open(file, encoding='utf8', errors='ignore') as f:
        for line in f:
            line = line.split(' ')
            word = line[0]
            vector = np.array([float(val) for val in line[1:]])
            model[word] = vector
                
    return model
