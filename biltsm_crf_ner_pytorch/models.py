import torch
import torch.nn as nn
from biltsm_crf_ner_pytorch.utils import argmax, log_sum_exp
from biltsm_crf_ner_pytorch.modules import charLSTM, BiLSTM, CRF

class Model(nn.Module)

    def __init__(self,
                 num_labels,
                 preprocessor,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_hidden_size=200,
                 char_lstm_hidden_size=25,
                 dropout=0.5,
                 embeddings=None,
                 use_char=True,
                 initial_vocab=None):
        self.model = None
        self.p = preprocessor

        self._num_labels = num_labels
        self._word_embedding_dim = word_embedding_dim
        self._char_embedding_dim = char_embedding_dim
        self._word_lstm_hidden_size = word_lstm_hidden_size
        self._char_lstm_hidden_size = char_lstm_hidden_size
        self._dropout = dropout
        self._pretrained_embeddings = embeddings

        self.initial_vocab = initial_vocab

        if use_char:
            self.char_embedder = charLSTM(char_embedding_dim=self._char_embedding_dim,
                                          char_lstm_hidden_size=self._char_lstm_hidden_size,
                                          dropout=self._dropout)
            self._word_embedding_dim += char_embedding_dim * 2

        self.seq_encoder = BiLSTM(num_labels=self._num_labels, vocab_size=self.,
                                  embedding_dim=self._word_embedding_dim, lstm_hidden_size=self._word_lstm_hidden_size,
                                  dropout=self._dropout, embeddings=self._pretrained_embeddings)
        self.tagger = CRF(num_labels=self._num_labels)
