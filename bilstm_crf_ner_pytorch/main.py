from logging import getLogger


from biltsm_crf_ner_pytorch.dataloader import DataLoader
#from biltsm_crf_ner_pytorch.preprocessing import Sequence

'''
class main(object):

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

    # def analyze(self, text, tokenizer=str.split):
    #    if not self.tagger:
    #        self.tagger = Tagger(self.model,
    #                            preprocessor=self.p,
    #                            tokenizer=tokenizer)
    #    return self.tagger.analyze(text)

    def save(self, weights_file, param_file, preprocessor_file):
        self.p.save(preprocessor_file)
        save_model(self.model, weights_file, params_file)

    @classmethod
    def load(cls, weights_file, params_file, preprocessor_file):
        self = cls()
        self.p = IndexTransformer.load(preprocessor_file)
        self.model = load_model(weights_file, params_file)

        return self
'''

if __name__ == '__main__':

    chemdner_train_link = '../../dataset/chemdner/full_type_data/conllform/train_conllform.txt'
    chemdner_dev_link = '../../dataset/chemdner/full_type_data/conllform/valid_conllform.txt'
    chemdner_test_link = '../../dataset/chemdner/full_type_data/conllform/test_conllform.txt'

    leader = DataLoader(anno_format='conll')

    x_train, y_train = leader.load_data(chemdner_train_link)
    x_dev, y_dev = leader.load_data(chemdner_dev_link)
    x_test, y_test = leader.load_data(chemdner_test_link)

    print(1e+12)
    #model = Sequence(use_char=True)
    #model.fit(x_train=x_train[:100], y_train=y_train[:100], epochs=1, batch_size=10, shuffle=True)