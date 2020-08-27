from logging import getLogger
import torch
import torch.nn as nn
import torch.optim as optim
import os

from bilstm_crf_ner_pytorch.dataloader import CorpusReader, filter_embeddings
from bilstm_crf_ner_pytorch.models import Model
from bilstm_crf_ner_pytorch.preprocessing import IndexTransformer
from bilstm_crf_ner_pytorch.trainer import Trainer

class main(object):

    def __init__(self,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_hidden_size=200,
                 char_lstm_hidden_size=25,
                 dropout=0.5,
                 embeddings=None,
                 use_char=True,
                 initial_vocab=None,
                 model_path=None,
                 preprocessor_path=None):
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
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, x_train, y_train, x_valid=None, y_valid=None,
            epochs=150, batch_size=32, shuffle=True, early_stop=5):
        preprocessor = IndexTransformer(initial_vocab=self.initial_vocab, use_char=self.use_char)
        preprocessor.fit(x_train, y_train)
        embeddings = filter_embeddings(self.embeddings, preprocessor._word_vocab.vocab, self.word_embedding_dim)

        model = Model(preprocessor=preprocessor,
                      embeddings=embeddings,
                      char_embedding_dim=self.char_embedding_dim,
                      char_lstm_hidden_size=self.char_lstm_hidden_size,
                      word_embedding_dim=self.word_embedding_dim,
                      word_lstm_hidden_size=self.word_lstm_hidden_size,
                      dropout=self.dropout,
                      use_char=self.use_char,
                      initial_vocab=self.initial_vocab,
                      device = self.device)

        optimizer = optim.SGD(model.parameters(), lr=0.015, weight_decay=1e-4)

        if self.device != 'cpu':
            model.to(self.device)

        trainer = Trainer(model,
                          optimizer=optimizer,
                          preprocessor=preprocessor,
                          use_char=self.use_char,
                          device=self.device)
        trainer.train(x_train, y_train, x_valid, y_valid,
                      epochs=epochs, batch_size=batch_size, shuffle=shuffle, early_stop=early_stop)
        self.p = preprocessor
        self.p.save(self.preprocessor_path)
        torch.save(trainer.best_model, self.model_path)

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


    #def analyze(self, text, tokenizer=str.split):
    #    if not self.tagger:
    #        self.tagger = Tagger(self.model,
    #                            preprocessor=self.p,
    #                            tokenizer=tokenizer)
    #    return self.tagger.analyze(text)

    @classmethod
    def load(cls, weights_file, params_file, preprocessor_file):
        self = cls()
        self.p = IndexTransformer.load(preprocessor_file)
        self.model = load_model(weights_file, params_file)

        return self


if __name__ == '__main__':

    chemdner_train_link = '../../dataset/chemdner/full_type_data/conllform/train_conllform.txt'
    chemdner_valid_link = '../../dataset/chemdner/full_type_data/conllform/valid_conllform.txt'
    chemdner_test_link = '../../dataset/chemdner/full_type_data/conllform/test_conllform.txt'

    leader = CorpusReader(anno_format='conll')

    x_train, y_train = leader.load_data(chemdner_train_link)
    x_valid, y_valid = leader.load_data(chemdner_valid_link)
    x_test, y_test = leader.load_data(chemdner_test_link)

    model_path = './best_model1_non_pretrained.model'
    processor_path = './processor.joblib'

    model = main(use_char=True, model_path=model_path, preprocessor_path=processor_path)
    model.fit(x_train, y_train, x_valid, y_valid, batch_size=100, shuffle=True, early_stop=10)
