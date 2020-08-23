# Load Library
import math
from collections import Counter
import random

import torch
import numpy as np
import torch.utils.data as data

class NERSequence(data.Dataset):

    def __init__(self, x, y, batch_size=1, preprocess=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.preprocess = preprocess

        
    def __getitem__(self, idx):
        #batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        #batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
        
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

class Dataset(data.Dataset):

    def __init__(self, text, label):
        self.text = text
        self.label = label
        self.num_data = len(label)

    def __getitem__(self, idx):
        return self.text[idx], self.label[idx]

    def __len__(self):
        return self.num_data


def collate_fn(data):

    def _merge_text(texts, labels):
        lengths = [len(s) for s in texts]
        padded_texts = torch.zeros(len(lengths), max(lengths)).long()
        padded_labels = torch.zeros(len(lengths), max(lengths)).long()
        for i, (text, label) in enumerate(zip(texts, labels)):
            end = lengths[i]
            padded_texts[i, :end] = torch.LongTensor(text[:end])
            padded_labels[i, :end] = torch.LongTensor(label[:end])
        return padded_texts, padded_labels, lengths

    # Reshape items
    texts, labels = zip(*data)
    # Convert to tensor
    padded_texts, padded_labels, lengths = _merge_text(texts, labels)

    # Sort by the sentence length to feed in pack_padded_sequence
    lengths, sort_index = torch.sort(torch.LongTensor(lengths), dim=0, descending=True)
    padded_texts = padded_texts[sort_index]
    padded_labels = padded_labels[sort_index]

    return padded_texts, padded_labels, lengths



def load_conll(path, encode):
    sentences, labels = [], []
    words, tags = [], []
    with open(path, encoding=encode) as f:
        for line in f:
            line = line.rstrip()
            if line:
                word, tag = line.split(' ')
                words.append(word)
                tags.append(tag)
            else:
                sentences.append(words)
                labels.append(tags)
                words, tags = [], []

    return sentences, labels

'''
def load_brat(path, encode):
    sents, labels = [], []
    words, tags = [], []
    # with open(self.path, encoding=self.encode) as f:

    return sents, labels

def load_pubchem(path, encode):
    sents, labels = [], []
    words, tags = [], []
    # with open(self.path, encoding=self.encode) as f:

    return sents, labels
'''

class DataLoader:
    def __init__(self, anno_format):
        assert anno_format in ['conll', 'brat', 'pubchem'], "The Annotation Format Does Not Correspond This Format"
        self.format = anno_format #{conll, brat, pubchem}

    def load_data(self, path, encode='utf-8'):
        assert path is not None, "The Data Path Is Not Allow Empty"
        if self.format == 'conll':
            sentences, labels = load_conll(path, encode)
        #elif self.format == 'brat':
        #    sentences, labels = load_brat(path, encode)
        #else: # <- 'pubchem'
        #    sentences, labels = load_pubchem(path, encode)
        else:
            sentences, labels = [], []
            
        return sentences, labels


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


def load_glove(file, model):
    with open(file, encoding='utf8', errors='ignore') as f:
        for line in f:
            line = line.split(' ')
            word = line[0]
            vector = np.array([float(val) for val in line[1:]])
            model[word] = vector
                
    return model
