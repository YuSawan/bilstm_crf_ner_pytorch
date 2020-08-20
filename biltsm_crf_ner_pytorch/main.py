from biltsm_crf_ner_pytorch.dataloader import DataLoader
from biltsm_crf_ner_pytorch.preprocessing import Sequence

if __name__ == '__main__':

    chemdner_train_link = '../dataset/chemdner/full_type_data/conllform/train_conllform.txt'
    chemdner_dev_link = '../dataset/chemdner/full_type_data/conllform/valid_conllform.txt'
    chemdner_test_link = '../dataset/chemdner/full_type_data/conllform/test_conllform.txt'

    leader = DataLoader(anno_format='conll')

    x_train, y_train = leader.load_data(chemdner_train_link)
    x_dev, y_dev = leader.load_data(chemdner_dev_link)
    x_test, y_test = leader.load_data(chemdner_test_link)

    model = Sequence(use_char=True)
    model.fit(x_train=x_train[:100], y_train=y_train[:100], epochs=1, batch_size=10, shuffle=True)