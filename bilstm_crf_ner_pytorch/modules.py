import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from biltsm_crf_ner_pytorch.utils import log_sum_exp

class charLSTM(nn.Module):
    def __init__(self,
                 vocab_size=None,
                 embedding_dim=25,
                 lstm_hidden_size=25,
                 dropout=0.5):
        super(charLSTM, self).__init__()
        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._lstm_hidden_size = lstm_hidden_size
        self._dropout = dropout

        self.char_embeddings = nn.Embedding(self._vocab_size, self._embedding_dim)
        self.char_lstm = nn.LSTM(input_size=self._embedding_dim, hidden_size=self._lstm_hidden_size,
                                 num_layers=1, dropout=self._dropout, batch_first=True, bidirectional=True)

    def init_hidden(self, batch_size):
        return {
            torch.zeros(2, batch_size, self._lstm_hidden_size),
            torch.zeros(2, batch_size, self._lstm_hidden_size)
        }

    def forward(self, x, num_character):
        total_length = x.size(1)
        x = pack(input=x, lengths=num_character, batch_first=True)

        init_hc = self.init_hidden(len(num_character))
        out, (hid_n, cell_n) = self.char_lstm(x, init_hc)

        # Unpack
        out, _ = unpack(
            out, batch_first=True, padding_value=0., total_length=total_length
        )

        return out, (hid_n, cell_n)


class BiLSTM(nn.Module):

    def __init__(self,
                 num_labels,
                 vocab_size,
                 embedding_dim=100,
                 lstm_hidden_size=200,
                 dropout=0.5,
                 embeddings=None,
                 gpu=False):
        super(BiLSTM, self).__init__()
        self._embedding_dim = embedding_dim
        self._lstm_hidden_size = lstm_hidden_size
        self._vocab_size = vocab_size
        self._dropout = dropout
        self._num_labels = num_labels
        self._gpu = gpu

        if embeddings is None:
            self.embeddings = nn.Embedding(self._vocab_size, self._embedding_dim)
        else:
            self.embeddings = nn.Embedding.from_pretrained(embeddings)
            self._embedding_dim = self.word_embeds.shape[0]


        self.lstm = nn.LSTM(input_size=self._embedding_dim, hidden_size=self._lstm_hidden_size // 2,
                            num_layers=1, dropout=self._dropout, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(self._lstm_hidden_size, self._num_labels)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(2, batch_size, self._lstm_hidden_size//2),
            torch.zeros(2, batch_size, self._lstm_hidden_size//2)/
        )

    def forword(self, x, num_sentence):
        total_length = x.size(1)
        x = pack(input=x, lengths=num_sentence, batch_first=True)

        init_hc = self.init_hidden(len(num_sentnece))
        out, (hid_n, cell_n) = self.lstm(x, init_hc)

        # Unpack
        out, _ = unpack(
            out, batch_first=True, padding_value=0., total_length=total_length
        )

        return out, (hid_n, cell_n)



class CRF(nn.Module):
    def __init__(self, num_labels, constraints=False, include_start_end=False):
        super(CRF, self).__init__()
        self._num_labels = num_labels

        # Transition[i, j] is the logit for transitioning from state i to state j
        self.transitions = nn.Parameter(torch.Tensor(num_labels, num_labels))

        # _constraint_mask indicates valid transitions (based on supplied constraints)
        # Include special start of sequence (num_labels + 1) and end of sequence tags (num_labels + 2)
        if constraints is None:
            constraint_mask = torch.Tensor(num_labels + 2, num_labels + 2).fill_(1.0)
        else:
            constraint_mask = torch.Tensor(num_labels + 2, num_labels + 2).fill_(0.0)
            for i, j in constraints:
                constraint_mask[i, j] = 1.0

        self._constrain_mask = nn.Parameter(constraint_mask, requires_grad=False)

        self.include_start_end_transtions = include_start_end
        if include_start_end:
            self.start_transtions = nn.Parameter(torch.Tensor(num_labels))
            self.end_transtions = nn.Parameter(torch.Tensor(num_labels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transtions:
            nn.init.normal_(self.start_transtions)
            nn.init.normal_(self.end_transtions)


    def _input_likelihood(self, logits, mask):
        batch_size, sequence_length, num_labels = logits.size()

        # Transpose batch size and sequence dimensions
        mask = mask.transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        if self.include_start_end_transtions:
            alpha = self.start_transtions.view(1, num_labels) + logits[0]
        else:
            alpha = logits[0]

        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so broadcast along the current tag axis
            emit_scores = logits[i].view(batch_size, 1, num_labels)

            # Transition scores are (current_tag, next_tag) so broadcast along the instance axis
            transition_scores = self.transitions.view(1, num_labels, num_labels)

            # ALpha is for the current_tag so broadcast along the next tag axis
            broadcast_alpha = alpha.view(batch_size, num_labels, 1)

            # Add all the scores together and logexp over the current_tag dimension
            inner  = broadcast_alpha + emit_scores + transition_scores

            # In valid positions(mask

