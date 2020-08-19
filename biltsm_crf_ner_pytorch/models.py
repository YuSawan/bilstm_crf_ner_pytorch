import torch
import torch.nn as nn
from biltsm_crf_ner_pytorch.src import argmax, log_sum_exp


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
                gpu = False):
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
        
        #self.word_hidden = self.init_hidden(len(sentence))
        self.word_hidden = None
        word_embeddings = self.embeddings(sentence)
        
        if self._use_char:
            char_reps = []
            for char in char_sequence.permute(1, 0, 2):
                self.char_hidden = None
                char_embeddings = self.char_embeddings(char)
                char_lstm_out, self.char_hidden = self.char_lstm(char_embeddings, self.char_hidden)
                char_lstm_out = torch.cat([char_lstm_out[:, 0, self._char_lstm_hidden_size:], char_lstm_out[:, -1, :self._char_lstm_hidden_size]], dim=1)
                char_reps.append(char_lstm_out)
            
            word_embeddings = [torch.cat([word_emb, char_rep], dim=1) for word_emb, char_rep in zip(word_embeddings.permute(1,0,2), char_reps)]
            word_embeddings = torch.stack(word_embeddings).permute(1, 0, 2)
        
        word_lstm_out, self.word_hidden = self.lstm(word_embeddings, self.word_hidden)
        #print(word_lstm_out.shape)
        lstm_feats = self.hidden2tag(word_lstm_out)
        #print(lstm_feats.shape)
        
        return lstm_feats
    
    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        if self._gpu:
            score = score.cuda()
        
        for k in range(feats.size(0)):
            for i, feat in enumerate(feats[k]):
                if i+1 == len(feats[k]):
                    break
                score = score + self.transitions[tags[k][i+1], tags[k][i]] + feat[tags[k][i+1]]
        
        return score / feats.size(0)
            
    def _viterbi_decode(self, feats):
        path_score = 0
        
        
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