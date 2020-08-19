import numpy as np
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.optim as optim

from seqeval.metrics import precision_score, recall_score, f1_score

from src.dataloader import Vocabulary, filter_embeddings
from src.trainer import Trainer
from src.models import BiLSTMCRF


class IndexTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, lower=True, num_norm=True,
                use_char=True, initial_vocab=None):
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
        word_ids = [torch.from_numpy(np.array(self._word_vocab.doc2id(doc))) for doc in X]
        word_ids = pad_sequence(word_ids, batch_first=True)
        
        if self._use_char:
            char_ids = [[self._char_vocab.doc2id(w) for w in doc] for doc in X]
            char_ids = pad_nested_sequences(char_ids)
            features = [word_ids, char_ids]
        else:
            features = word_ids
        
        if y is not None:
            y = [torch.from_numpy(np.array(self._label_vocab.doc2id(doc))) for doc in y]
            y = pad_sequence(y, batch_first=True)
            #y = np.eye(self.label_size, dtype='uint8')[y]            
            #y = y if len(y.shape) == 3 else np.expand_dims(y, axis=0)
            return features, y
        else:
            return features
        
    def fit_transform(self, X, y=None, **params):
        return self.fit(X, y).transform(X, y)
    
    def inverse_transform(self, y, lengths=None):
        y = np.argmax(y, -1)
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


class Sequence(object):
    
    def __init__(self,
                word_embedding_dim=100,
                char_embedding_dim=25,
                word_lstm_hidden_size=200,
                char_lstm_hidden_size=25,
                dropout=0.5,
                embeddings=None,
                use_char=True,
                initial_vocab=None):
        self.model = None
        self.p = None
        self.tagger = None
        
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.word_lstm_hidden_size = word_lstm_hidden_size
        self.char_lstm_hidden_size = char_lstm_hidden_size
        self.dropout = dropout
        self.embeddings = embeddings
        self.use_char = use_char
        self.initial_vocab = initial_vocab
        self.gpu = torch.cuda.is_available()
                
    def fit(self, x_train, y_train, x_valid=None, y_valid=None,
           epochs=150, batch_size=32, shuffle=True):
        p = IndexTransformer(initial_vocab=self.initial_vocab, use_char=self.use_char)
        p.fit(x_train, y_train)
        embeddings = filter_embeddings(self.embeddings, p._word_vocab.vocab, self.word_embedding_dim)
        
        model = BiLSTMCRF(char_vocab_size=p.char_vocab_size,
                         vocab_size=p.word_vocab_size,
                         num_labels=p.label_size,
                         embedding_dim=self.word_embedding_dim,
                         char_embedding_dim=self.char_embedding_dim,
                         lstm_hidden_size=self.word_lstm_hidden_size,
                         char_lstm_hidden_size=self.char_lstm_hidden_size,
                         dropout=self.dropout,
                         embeddings=embeddings,
                         use_char=self.use_char,
                         gpu=self.gpu)
        
        
        optimizer = optim.SGD(model.parameters(), lr=0.015, weight_decay=1e-4)
        
        
        trainer = Trainer(model, optimizer, preprocessor=p, use_char=self.use_char, gpu=self.gpu)
        trainer.train(x_train, y_train, x_valid, y_valid,
                     epochs=epochs, batch_size=batch_size, shuffle=shuffle)
        
        self.p = p
        self.model = model

    def predict(self, x_test):
        lengths = map(len, x_test)
        x_test = self.p.transform(x_test)
        
        if self.model:
            lengths = map(len, x_test)
            x_test = self.p.transform(x_test)
            
            y_pred = self.model.predict(x_test)
            y_pred = self.p.inverse_transform(y_pred)
            return y_pred
        else:
            raise OSError('Could not find a model. Call load(dir_path).')
        
        return self.p.inverse_transform(x_test)
        
    def score(self, x_test, y_test):
        if self.model:
            x_test = self.p.transform(x_test)
            lengths = map(len, y_test)
            y_pred = self.model.predict(x_test)
            y_pred = self.p.inverse_transform(y_pred)
            score = f1_score(y_test, y_pred)
            return score
        else:
            raise OSError('Could not find a model. Call load(dir_path).')
            
    def analyze(self, text, tokenizer=str.split):
        if not self.tagger:
            self.tagger = Tagger(self.model,
                                preprocessor=self.p,
                                tokenizer=tokenizer)
        return self.tagger.analyze(text)
    
    def save(self, weights_file, param_file, preprocessor_file):
        self.p.save(preprocessor_file)
        save_model(self.model, weights_file, params_file)
        
    @classmethod
    def load(cls, weights_file, params_file, preprocessor_file):
        self = cls()
        self.p = IndexTransformer.load(preprocessor_file)
        self.model = load_model(weights_file, params_file)
        
        return self
