from biltsm_crf_ner_pytorch.src import DataLoader
from biltsm_crf_ner_pytorch.src import Sequence

if __name__ == '__main__':
    chemdner_train_link = '../datasets/full_type_data/conllform/train_conllform.txt'
    chemdner_dev_link = '../datasets/full_type_data/conllform/valid_conllform.txt'
    chemdner_test_link = '../datasets/full_type_data/conllform/test_conllform.txt'

    leader = DataLoader(anno_format='conll')

    x_train, y_train = leader.load_data(chemdner_train_link)
    x_dev, y_dev = leader.load_data(chemdner_dev_link)
    x_test, y_test = leader.load_data(chemdner_test_link)

    model = Sequence(use_char=True)
    model.fit(x_train=x_train, y_train=y_train, epochs=1, batch_size=10, shuffle=True)