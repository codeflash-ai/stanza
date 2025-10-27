"""
CRF loss and viterbi decoding.
"""

import math
from numbers import Number
import numpy as np
import torch
from torch import nn
import torch.nn.init as init

class CRFLoss(nn.Module):
    """
    Calculate log-space crf loss, given unary potentials, a transition matrix
    and gold tag sequences.
    """
    def __init__(self, num_tag, batch_average=True):
        super().__init__()
        self._transitions = nn.Parameter(torch.zeros(num_tag, num_tag))
        self._batch_average = batch_average # if not batch average, average on all tokens

    def forward(self, inputs, masks, tag_indices):
        """
        inputs: batch_size x seq_len x num_tags
        masks: batch_size x seq_len
        tag_indices: batch_size x seq_len
        
        @return:
            loss: CRF negative log likelihood on all instances.
            transitions: the transition matrix
        """
        # TODO: handle <start> and <end> tags
        input_bs, input_sl, input_nc = inputs.size()
        unary_scores = self.crf_unary_score(inputs, masks, tag_indices, input_bs, input_sl, input_nc)
        binary_scores = self.crf_binary_score(inputs, masks, tag_indices, input_bs, input_sl, input_nc)
        log_norm = self.crf_log_norm(inputs, masks, tag_indices)
        log_likelihood = unary_scores + binary_scores - log_norm # batch_size
        loss = torch.sum(-log_likelihood)
        if self._batch_average:
            loss = loss / input_bs
        else:
            total = masks.eq(0).sum()
            loss = loss / (total + 1e-8)
        return loss, self._transitions

    def crf_unary_score(self, inputs, masks, tag_indices, input_bs, input_sl, input_nc):
        """
        @return:
            unary_scores: batch_size
        """
        # Precompute arange once for efficiency
        arange_sl = torch.arange(input_sl, device=tag_indices.device).long().unsqueeze(0)
        flat_tag_indices = tag_indices + arange_sl * input_nc

        flat_inputs = inputs.view(input_bs, -1)
        unary_scores = torch.gather(flat_inputs, 1, flat_tag_indices).view(input_bs, -1)
        unary_scores.masked_fill_(masks, 0)
        return unary_scores.sum(dim=1)
    
    def crf_binary_score(self, inputs, masks, tag_indices, input_bs, input_sl, input_nc):
        """
        @return:
            binary_scores: batch_size
        """
        nt = tag_indices.size(-1) - 1
        start_indices = tag_indices[:, :nt]
        end_indices = tag_indices[:, 1:]
        flat_transition_indices = (start_indices * input_nc + end_indices).reshape(-1)
        flat_transition_matrix = self._transitions.reshape(-1)
        binary_scores = torch.gather(flat_transition_matrix, 0, flat_transition_indices)\
                .view(input_bs, -1)
        score_masks = masks[:, 1:]
        binary_scores.masked_fill_(score_masks, 0)
        return binary_scores.sum(dim=1)

    def crf_log_norm(self, inputs, masks, tag_indices):
        """
        Calculate the CRF partition in log space for each instance, following:
            http://www.cs.columbia.edu/~mcollins/fb.pdf
        @return:
            log_norm: batch_size
        """
        start_inputs = inputs[:,0,:] # bs x nc
        rest_inputs = inputs[:,1:,:]
        rest_masks = masks[:,1:]

        alphas = start_inputs # bs x nc
        trans = self._transitions.unsqueeze(0) # 1 x nc x nc

        # Optimize internal loop using batch and vectorization for memory and runtime
        rest_seq_len = rest_inputs.size(1)
        batch_size, num_tag = alphas.size()
        for i in range(rest_seq_len):
            # bs x nc x nc = [alphas]+[trans]
            transition_scores = alphas.unsqueeze(2) + trans
            # Vectorize log_sum_exp for all batches
            # new_alphas shape: bs x nc
            new_alphas = rest_inputs[:,i,:] + log_sum_exp(transition_scores, dim=1)

            # Masks: only update unmasked positions
            m = rest_masks[:,i].unsqueeze(1) # bs x 1
            # Use torch.where instead of masked_scatter_ for faster masking
            # If mask == 1 (pad), keep previous alpha, else update
            alphas = torch.where(m, alphas, new_alphas)

        log_norm = log_sum_exp(alphas, dim=1)
        all_masked = torch.all(masks, dim=1)
        log_norm = log_norm * torch.logical_not(all_masked)
        return log_norm

def viterbi_decode(scores, transition_params):
    """
    Decode a tag sequence with viterbi algorithm.
    scores: seq_len x num_tags (numpy array)
    transition_params: num_tags x num_tags (numpy array)
    @return:
        viterbi: a list of tag ids with highest score
        viterbi_score: the highest score
    """
    trellis = np.zeros_like(scores)
    backpointers = np.zeros_like(scores, dtype=np.int32)
    trellis[0] = scores[0]

    for t in range(1, scores.shape[0]):
        v = np.expand_dims(trellis[t-1], 1) + transition_params
        trellis[t] = scores[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)
