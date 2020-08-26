import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import torch


from bilstm_crf_ner_pytorch.dataloader import Vocabulary


class IndexTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, lower=True, num_norm=True, use_char=True, initial_vocab=None):
        self._num_norm = num_norm
        self._use_char = use_char
        self._word_vocab = Vocabulary(lower=lower)
        self._char_vocab = Vocabulary(lower=False)
        self._label_vocab = Vocabulary(lower=False, unk_token=False)
        
        if initial_vocab:
            self._word_vocab.add_documents([initial_vocab])
            self._char_vocab.add_documents(initial_vocab)
            
    def fit(self, X, y):
        self._word_vocab.add_documents(X)
        self._label_vocab.add_documents(y)
        if self._use_char:
            for doc in X:
                self._char_vocab.add_documents(doc)
                
        self._word_vocab.build()
        self._char_vocab.build()
        self._label_vocab.build()
        
        return self
        
    def transform(self, X, y=None):
        word_ids = self._word_vocab.doc2id(X)

        if self._use_char:
            char_ids = [self._char_vocab.doc2id(w) for w in X]
            features = (word_ids, char_ids)
        else:
            features = word_ids

        if y is not None:
            y = self._label_vocab.doc2id(y)
            return features, y
        else:
            return features

    def fit_transform(self, X, y=None, **params):
        return self.fit(X, y).transform(X, y)
    
    def inverse_transform(self, y, lengths=None):
        inverse_y = [self._label_vocab.id2doc(ids) for ids in y]
        if lengths is not None:
            inverse_y = [iy[:l] for iy, l in zip(inverse_y, lengths)]
            
        return inverse_y
    
    @property
    def word_vocab_size(self):
        return len(self._word_vocab)
    
    @property
    def char_vocab_size(self):
        return len(self._char_vocab)
    
    @property
    def label_size(self):
        return len(self._label_vocab)
    
    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)
        
        return p
