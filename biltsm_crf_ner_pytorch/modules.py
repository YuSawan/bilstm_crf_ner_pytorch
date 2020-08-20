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


class BiLSTMCRF(nn.Module):
    def __init__(self,
                 num_labels,
                 vocab_size,
                 embedding_dim=100,
                 lstm_hidden_size=200,
                 char_vocab_size=None,
                 char_embedding_dim=25,
                 char_lstm_hidden_size=25,
                 dropout=0.5,
                 embeddings=None,
                 use_char=True,
                 gpu=False):
        super(BiLSTMCRF, self).__init__()
        self._char_vocab_size = char_vocab_size
        self._char_embedding_dim = char_embedding_dim
        self._char_lstm_hidden_size = char_lstm_hidden_size
        self._embedding_dim = embedding_dim
        self._lstm_hidden_size = lstm_hidden_size
        self._vocab_size = vocab_size
        self._use_char = use_char
        self._dropout = dropout
        self._num_labels = num_labels
        self._gpu = gpu

        if embeddings is None:
            self.embeddings = nn.Embedding(self._vocab_size, self._embedding_dim)
        else:
            self.embeddings = nn.Embedding.from_pretrained(embeddings)
            self._embedding_dim = self.word_embeds.shape[0]

        if self._use_char:
            self.char_embeddings = nn.Embedding(self._char_vocab_size, self._char_embedding_dim)
            self.char_lstm = nn.LSTM(input_size=self._char_embedding_dim,
                                     hidden_size=self._char_lstm_hidden_size,
                                     num_layers=1,
                                     dropout=self._dropout,
                                     batch_first=True,
                                     bidirectional=True)
            self._embedding_dim += self._char_embedding_dim * 2

        self.lstm = nn.LSTM(input_size=self._embedding_dim,
                            hidden_size=self._lstm_hidden_size // 2,
                            num_layers=1,
                            dropout=self._dropout,
                            batch_first=True,
                            bidirectional=True)
        self.hidden2tag = nn.Linear(self._lstm_hidden_size, self._num_labels)
        self.transitions = nn.Parameter(torch.randn(self._num_labels, self._num_labels))

        if self._gpu:
            if self._use_char:
                self.char_embeddings = self.char_embeddings.cuda()
                self.char_lstm = self.char_lstm.cuda()
            self.embeddings = self.embeddings.cuda()
            self.lstm = self.lstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.transitions = nn.Parameter(torch.randn(self._num_labels, self._num_labels).cuda())

    def _forword_alg(self, feats):
        init_alphas = torch.full((1, self._num_labels), -1000)
        if self._gpu:
            init_alphas = init_alphas.cuda()

        forward_var = init_alphas
        alpha = 0

        for i in range(feats.size(0)):
            for feat in feats[i]:
                alphas_t = []
                for next_tag in range(self._num_labels):
                    emit_score = feat[next_tag].view(1, -1).expand(1, self._num_labels)
                    trans_score = self.transitions[next_tag].view(1, -1)

                    next_tag_var = forward_var + trans_score + emit_score
                    alphas_t.append(log_sum_exp(next_tag_var).view(1))
                forward_var = torch.cat(alphas_t).view(1, -1)

            alpha += log_sum_exp(forward_var)

        return alpha / feats.size(0)

    def _get_lstm_features(self, sentence, char_sequence=None):

        # self.word_hidden = self.init_hidden(len(sentence))
        self.word_hidden = None
        word_embeddings = self.embeddings(sentence)

        if self._use_char:
            char_reps = []
            for char in char_sequence.permute(1, 0, 2):
                self.char_hidden = None
                char_embeddings = self.char_embeddings(char)
                char_lstm_out, self.char_hidden = self.char_lstm(char_embeddings, self.char_hidden)
                char_lstm_out = torch.cat([char_lstm_out[:, 0, self._char_lstm_hidden_size:],
                                           char_lstm_out[:, -1, :self._char_lstm_hidden_size]], dim=1)
                char_reps.append(char_lstm_out)

            word_embeddings = [torch.cat([word_emb, char_rep], dim=1) for word_emb, char_rep in
                               zip(word_embeddings.permute(1, 0, 2), char_reps)]
            word_embeddings = torch.stack(word_embeddings).permute(1, 0, 2)

        word_lstm_out, self.word_hidden = self.lstm(word_embeddings, self.word_hidden)
        # print(word_lstm_out.shape)
        lstm_feats = self.hidden2tag(word_lstm_out)
        # print(lstm_feats.shape)

        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        if self._gpu:
            score = score.cuda()

        for k in range(feats.size(0)):
            for i, feat in enumerate(feats[k]):
                if i + 1 == len(feats[k]):
                    break
                score = score + self.transitions[tags[k][i + 1], tags[k][i]] + feat[tags[k][i + 1]]

        return score / feats.size(0)

    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self._num_labels), -10000.)
        forward_var = init_vvars
        for feat in feats:
            bbtrs_t = []
            viterbivars_t = []

            for next_tag in range(self._num_labels):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bbtrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bbtrs_t)

        path_score = forward_var

        best_path = [best_tag_id]
        for bbtrs_t in reversed(backpointers):
            best_tag_id = bbtrs_t[best_tag_id]
            best_path.append(best_tag_id)

        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags, char_sequence=None):
        if self._use_char:
            feats = self._get_lstm_features(sentence, char_sequence)
        else:
            feats = self._get_lstm_features(sentence)

        forward_score = self._forword_alg(feats)
        gold_score = self._score_sentence(feats, tags)

        return forward_score - gold_score

    def forward(self, sentence, char_sequence=None):
        if self._use_char:
            lstm_feats = self._get_lstm_features(sentence, char_sequence)
        else:
            lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
