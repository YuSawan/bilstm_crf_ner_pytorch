import torch
import torch.nn as nn
from bilstm_crf_ner_pytorch.modules import charLSTM, BiLSTM, CRF


class Model(nn.Module):

    def __init__(self,
                 preprocessor,
                 char_embedding_dim=25,
                 char_lstm_hidden_size=25,
                 word_embedding_dim=100,
                 word_lstm_hidden_size=200,
                 dropout=0.5,
                 embeddings=None,
                 use_char=True,
                 initial_vocab=None):
        super(Model, self).__init__()
        self.model = None,
        self.p = preprocessor
        self._char_embedding_dim = char_embedding_dim
        self._char_lstm_hidden_size = char_lstm_hidden_size
        self._word_embedding_dim = word_embedding_dim
        self._word_lstm_hidden_size = word_lstm_hidden_size
        self._dropout = dropout
        self._pretrained_embeddings = embeddings
        self.initial_vocab = initial_vocab
        self._use_char = use_char

        if embeddings is None:
            self.word_embedder = nn.Embedding(self.p.word_vocab_size, self._word_embedding_dim)
        else:
            self._pretrained_embeddings = nn.Embedding.from_pretrained(embeddings)
            self._word_embedding_dim = self.word_embeds.shape[0]

        if use_char:
            self.char_embedder = charLSTM(vocab_size=self.p.char_vocab_size,
                                          embedding_dim=self.char_embedding_dim,
                                          lstm_hidden_size=self._char_lstm_hidden_size,
                                          dropout=self._dropout)
            self._word_embedding_dim += char_embedding_dim * 2

        self.seq_encoder = BiLSTM(num_labels=self.p.label_size, vocab_size=self.p.word_vocab_size,
                                  embedding_dim=self._word_embedding_dim, lstm_hidden_size=self._word_lstm_hidden_size,
                                  dropout=self._dropout, embeddings=self._pretrained_embeddings)
        self.hidden2tag = nn.Linear(self._word_lstm_hidden_size, self.p.label_size)
        self.tagger = CRF(num_labels=self.p.label_size)

    def decode_tags(self, batch):
        tok_tensor, tag_tensor, lengths = batch
        tok_embs = self.word_embedder(tok_tensor)

        if self._use_char:
            char_embs = self.char_embedder(tok_tensor)
            tok_embs = torch.cat((tok_embs, char_embs), dim=0)

        seq_embs, _ = self.seq_encoder(tok_embs, lengths)
        outputs = self.hidden2tag(seq_embs)
        tag_seq = self.tagger.viterbi_tags(outputs)

        return tag_seq, tag_tensor

    def forward(self, batch):
        tok_tensor, tag_tensor, lengths = batch
        tok_embs = self.word_embedder(tok_tensor)

        if self._use_char:
            char_embs = self.char_embedder(tok_tensor)
            tok_embs = torch.cat((tok_embs, char_embs), dim=0)

        seq_embs, _ = self.seq_encoder(tok_embs, lengths)
        outputs = self.hidden2tag(seq_embs)
        loss = self.tagger(outputs, tag_tensor)

        return loss
