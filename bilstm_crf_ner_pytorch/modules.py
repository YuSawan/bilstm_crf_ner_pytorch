import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from bilstm_crf_ner_pytorch.utils import log_sum_exp, viterbi_decode


class charLSTM(nn.Module):
    def __init__(self,
                 vocab_size=None,
                 embedding_dim=25,
                 lstm_hidden_size=25,
                 dropout=0.5,
                 device='cpu'):
        super(charLSTM, self).__init__()
        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._lstm_hidden_size = lstm_hidden_size
        self._dropout = dropout

        self.dropout = nn.Dropout(p=self._dropout)
        self.char_embeddings = nn.Embedding(self._vocab_size, self._embedding_dim)
        self.char_lstm = nn.LSTM(input_size=self._embedding_dim, hidden_size=self._lstm_hidden_size,
                                 num_layers=1, dropout=self._dropout, batch_first=True, bidirectional=True)


    def get_last_hiddens(self, x_char, num_character):
        batch_size =  x_char.size(0)
        char_embeds = self.char_embeddings(x_char)
        char_embeds = self.dropout(char_embeds)
        char_hidden = None
        char_embeds = pack(input=char_embeds, lengths=num_character, batch_first=True)
        out, (hid_n, cell_n) = self.char_lstm(char_embeds, char_hidden)
        return hid_n.transpose(1, 0).contiguous().view(batch_size, -1)


    def get_all_hiddens(self, x_char, num_character):
        char_embeds = self.char_embeddings(x_char)
        char_hidden = None
        char_embeds = pack(input=char_embeds, lengths=num_character, batch_first=True)
        out, (hid_n, cell_n)  = self.char_lstm(char_embeds, char_hidden)
        out, _ = unpack(out, batch_first=True, padding_value=0.)
        return out, (hid_n, cell_n)

    def forward(self, x_char, num_character):
        return self.get_all_hiddens(x_char, num_character)


class BiLSTM(nn.Module):

    def __init__(self,
                 num_labels,
                 vocab_size,
                 embedding_dim=100,
                 lstm_hidden_size=200,
                 dropout=0.5,
                 embeddings=None,
                 device='cpu'):
        super(BiLSTM, self).__init__()
        self._embedding_dim = embedding_dim
        self._lstm_hidden_size = lstm_hidden_size
        self._vocab_size = vocab_size
        self._dropout = dropout
        self._num_labels = num_labels
        self._device = device

        self.lstm = nn.LSTM(input_size=self._embedding_dim, hidden_size=self._lstm_hidden_size // 2,
                            num_layers=1, dropout=self._dropout, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(self._lstm_hidden_size, self._num_labels)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(2, batch_size, self._lstm_hidden_size//2).to(self._device),
            torch.zeros(2, batch_size, self._lstm_hidden_size//2).to(self._device)
        )

    def forward(self, x, num_sentence):
        total_length = x.size(1)
        x = pack(input=x, lengths=num_sentence, batch_first=True)

        init_hc = self.init_hidden(len(num_sentence))
        out, (hid_n, cell_n) = self.lstm(x, init_hc)

        # Unpack
        out, _ = unpack(
            out, batch_first=True, padding_value=0., total_length=total_length
        )

        return out, (hid_n, cell_n)


class CRF(nn.Module):
    def __init__(self, num_labels, constraints=None, include_start_end=False, device='cpu'):
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
            self.start_transtions = nn.Parameter(torch.Tensor(num_labels), requires_grad=False)
            self.end_transtions = nn.Parameter(torch.Tensor(num_labels), requires_grad=False)

        self.device = device

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
            inner = broadcast_alpha + emit_scores + transition_scores

            # In valid positions(mask == True) we want to take the logsumexp over the current_tag dimension
            # of 'inner'. Otherwise (mask=False) we want to retain the previous alpha
            alpha = log_sum_exp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (~mask[i]).view(batch_size, 1)

        # Every sequence needs to end with a transition to the stop_tag
        if self.include_start_end_transtions:
            stops = alpha + self.end_transtions.view(1, num_labels)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_labels dim, result is (batch_size,)
        return log_sum_exp(stops)

    def _joint_likelihood(self, logits, tags, mask):
        batch_size, sequence_length, _ = logits.data.shape

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag tag in each input
        if self.include_start_end_transtions:
            score = self.start_transtions.index_select(0, tags[0])
        else:
            score = 0.0

        # Add up the scores for the observed transitioning from current_tag to next_tag
        for i in range(sequence_length - 1):
            # Each is shape (batch_size, )
            current_tag, next_tag = tags[i], tags[i + 1]

            # The scores for transitioning from current_tag to next_tag
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]

            # The score for using current_tag
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)

            # Include transition score if next element is unmasked
            # input_score if this element is unmasked
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        # Transition from last state to "stop" state. To start with, we need to find the last tag for each instance
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        # Compute score of transitioning to 'stop_tag' from each 'last tags'
        if self.include_start_end_transtions:
            last_transition_score = self.end_transtions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        # Add the last input if it's not masked
        last_inputs = logits[-1]
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size, )

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def forward(self, inputs, tags, mask=None):
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.bool).to(self.device)
        else:
            mask = mask.to(torch.bool)
        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        return torch.sum(log_denominator - log_numerator)

    def viterbi_tags(self, logits, mask=None, top_k=None):
        if mask is None:
            mask = torch.ones(*logits.shape[:2], dtype=torch.bool, device=logits.device)

        if top_k is None:
            top_k = 1
            flatten_output = True
        else:
            flatten_output = False

        _, max_seq_length, num_tags = logits.size()

        # Get the tensors our of the variables
        logits, mask = logits.data, mask.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-1000.0)

        # Apply transition constraints
        constrained_transitions = self.transitions * self._constrain_mask[:num_tags, :num_tags]\
                                  + -1000.0 * (1 - self._constrain_mask[:num_tags, :num_tags])
        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_start_end_transtions:
            transitions[
                start_tag, :num_tags
            ] = self.start_transtions.detach() * self._constrain_mask[
                start_tag, :num_tags
            ].data + -10000.0 * (
                1 - self._constrain_mask[start_tag, :num_tags].detach()
            )
            transitions[:num_tags, end_tag] = self.end_transtions.detach() * self._constrain_mask[
                :num_tags, end_tag
            ].data + -10000.0 * (
                1 - self._constrain_mask[:num_tags, end_tag].detach()
            )
        else:
            transitions[start_tag, :num_tags] = -10000.0 * (
                1 - self._constrain_mask[start_tag, :num_tags].detach()
            )
            transitions[:num_tags, end_tag] = -10000.0 * (
                1 - self._constrain_mask[:num_tags, end_tag].detach()
            )

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)

        for prediction, prediction_mask in zip(logits, mask):
            mask_indices = prediction_mask.nonzero().squeeze()
            masked_prediction = torch.index_select(prediction, 0, mask_indices)
            sequence_length = masked_prediction.shape[0]

            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.0)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.0
            # At steps 1, ..., sequence_length we junst use the incoming prediction
            tag_sequence[1: (sequence_length + 1), :num_tags] = masked_prediction
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, end_tag] = 0.0

            # We pass the tags and the transitions to 'viterbi_decode'
            viterbi_paths, viterbi_scores = viterbi_decode(
                tag_sequence=tag_sequence[: (sequence_length + 2)],
                transition_matrix=transitions,
                top_k=top_k
            )
            top_k_paths = []
            for viterbi_path, viterbi_score in zip(viterbi_paths, viterbi_scores):
                # Get rid of START and END sentinels and append
                viterbi_path = viterbi_path[1:-1]
                top_k_paths.append((viterbi_path, viterbi_score.item()))
            best_paths.append(top_k_paths)

        if flatten_output:
            return [top_k_paths[0] for top_k_paths in best_paths]

        return best_paths
