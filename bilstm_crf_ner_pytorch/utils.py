import torch
import math
import json
from logging import getLogger

logger = getLogger(__name__)

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(tensor, dim=-1, keepdim=False):
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)

    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def viterbi_decode(tag_sequence, transition_matrix, tag_observations=None,
                   allowed_start_transitions=None, allowed_end_transitions=None, top_k=None):
    if top_k is None:
        top_k = 1
        flatten_output = True
    elif top_k >= 1:
        flatten_output = False
    else:
        raise ValueError(f"top_k must be either None or an integer >= 1. Instead received {top_k}")

    sequence_length, num_tags = list(tag_sequence.size())

    has_start_end_restriction = (
        allowed_end_transitions is not None or allowed_start_transitions is not None
    )

    if has_start_end_restriction:

        if allowed_end_transitions is None:
            allowed_end_transitions = torch.zeros(num_tags)
        if allowed_start_transitions is None:
            allowed_start_transitions = torch.zeros(num_tags)

        num_tags = num_tags + 2
        new_transition_matrix = torch.zeros(num_tags, num_tags)
        new_transition_matrix[:-2, :-2] = transition_matrix

        # Start and end transitions are fully defined, but cannot transition between each other

        allowed_start_transitions = torch.cat(
            [allowed_start_transitions, torch.tensor([-math.inf, -math.inf])]
        )
        allowed_end_transitions = torch.cat(
            [allowed_end_transitions, torch.tensor([-math.inf, -math.inf])]
        )

        # First define how we may transition FROM the start and end tags
        new_transition_matrix[-2, :] = allowed_start_transitions
        # We cannot transition from the end tag to any tag
        new_transition_matrix[-1, :] = -math.inf

        new_transition_matrix[:, -1] = allowed_end_transitions
        # We cannot transition from the start tag to any tag
        new_transition_matrix[:, -2] = -math.inf

        transition_matrix = new_transition_matrix

    if tag_observations is None:
        tag_observations = [-1 for _ in range(sequence_length)]

    if has_start_end_restriction:
        tag_observations = [num_tags - 2] + tag_observations + [num_tags - 1]
        zero_sentinel = torch.zeros(1, num_tags)
        extra_tags_sentinel = torch.ones(sequence_length, 2) * -math.inf
        tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
        tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)
        sequence_length = tag_sequence.size(0)

    path_scores = []
    path_indices = []

    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.0
        path_scores.append(one_hot.unsqueeze(0))
    else:
        path_scores.append(tag_sequence[0, :].unsqueeze(0))

    # Evaluate the scores for all possible paths
    for timestep in range(1, sequence_length):
        summed_potentials = path_scores[timestep-1].unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags)

        max_k = min(summed_potentials.size()[0], top_k)
        scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)

        observation = tag_observations[timestep]

        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.0
            path_scores.append(one_hot.unsqueeze(0))
        else:
            path_scores.append(tag_sequence[timestep, :] + scores)
        path_indices.append(paths.squeeze())

    # Construct the most unlikely sequence backwards
    path_scores_v = path_scores[-1].view(-1)
    max_k = min(path_scores_v.size()[0], top_k)
    viterbi_scores, best_paths = torch.topk(path_scores_v, k=max_k, dim=0)
    viterbi_paths = []
    for i in range(max_k):
        viterbi_path = [best_paths[i]]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
        # Reverse the backward path
        viterbi_path.reverse()

        if has_start_end_restriction:
            viterbi_path = viterbi_path[1:-1]

        # Viterbi paths uses (num_labels * n_permutations) nodes: therefore, we need to mpdulo
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_paths.append(viterbi_path)

        if flatten_output:
            return viterbi_paths[0], viterbi_scores[0]

        return viterbi_paths, viterbi_scores


def get_entities(seq, preprocessor, lengths):
    seq = preprocessor.inverse_transform(seq, lengths=lengths)
    seq = [item for sublist in seq for item in sublist]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq+['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag):
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag):
    chunk_end = False

    if prev_tag == 'E' or prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag):
    chunk_start = False

    if tag == 'B' or tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    return chunk_start
