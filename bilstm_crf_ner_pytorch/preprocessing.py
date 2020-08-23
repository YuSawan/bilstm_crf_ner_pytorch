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
        X = [self._word_vocab.doc2id(doc) for doc in X]

        if y is not None:
            y = [self._label_vocab.doc2id(doc) for doc in y]
            return X, y
        else:
            return X
        
    def fit_transform(self, X, y=None, **params):
        return self.fit(X, y).transform(X, y)
    
    def inverse_transform(self, y, lengths=None):
        print(y)
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


def pad_nested_sequences(sequences, dtype='int64'):
    max_sent_len = 0
    max_word_len = 0
    for sent in sequences:
        max_sent_len = max(len(sent), max_sent_len)
        for word in sent:
            max_word_len = max(len(word), max_word_len)
            
    x = np.zeros((len(sequences), max_sent_len, max_word_len)).astype(dtype)
    for i, sent in enumerate(sequences):
        for j, word in enumerate(sent):
            x[i, j, :len(word)] = word
            
    return torch.from_numpy(x)
