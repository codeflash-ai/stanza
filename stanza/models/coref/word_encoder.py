""" Describes WordEncoder. Extracts mention vectors from bert-encoded text.
"""

from typing import Tuple

import torch

from stanza.models.coref.config import Config
from stanza.models.coref.const import Doc


class WordEncoder(torch.nn.Module):  # pylint: disable=too-many-instance-attributes
    """ Receives bert contextual embeddings of a text, extracts all the
    possible mentions in that text. """

    def __init__(self, features: int, config: Config):
        """
        Args:
            features (int): the number of featues in the input embeddings
            config (Config): the configuration of the current session
        """
        super().__init__()
        self.attn = torch.nn.Linear(in_features=features, out_features=1)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

    @property
    def device(self) -> torch.device:
        """ A workaround to get current device (which is assumed to be the
        device of the first parameter of one of the submodules) """
        return next(self.attn.parameters()).device

    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                doc: Doc,
                x: torch.Tensor,
                ) -> Tuple[torch.Tensor, ...]:
        """
        Extracts word representations from text.

        Args:
            doc: the document data
            x: a tensor containing bert output, shape (n_subtokens, bert_dim)

        Returns:
            words: a Tensor of shape [n_words, mention_emb];
                mention representations
            cluster_ids: tensor of shape [n_words], containing cluster indices
                for each word. Non-coreferent words have cluster id of zero.
        """
        # Directly create tensor in correct dtype and device to avoid extra copy
        word_boundaries = torch.as_tensor(doc["word2subword"], device=self.device, dtype=torch.long)
        starts = word_boundaries[:, 0]
        ends = word_boundaries[:, 1]

        # [n_mentions, features]
        words_attn = self._attn_scores(x, starts, ends)
        words = torch.matmul(words_attn, x)

        words = self.dropout(words)

        return (words, self._cluster_ids(doc))

    def _attn_scores(self,
                     bert_out: torch.Tensor,
                     word_starts: torch.Tensor,
                     word_ends: torch.Tensor) -> torch.Tensor:
        """ Calculates attention scores for each of the mentions.

        Args:
            bert_out (torch.Tensor): [n_subwords, bert_emb], bert embeddings
                for each of the subwords in the document
            word_starts (torch.Tensor): [n_words], start indices of words
            word_ends (torch.Tensor): [n_words], end indices of words

        Returns:
            torch.Tensor: [description]
        """
        n_subtokens = bert_out.size(0)
        n_words = word_starts.size(0)

        # Efficient mask generation using broadcasting (avoids expand and mul)
        ar = torch.arange(n_subtokens, device=self.device)
        attn_mask = (ar.unsqueeze(0) >= word_starts.unsqueeze(1)) & (ar.unsqueeze(0) < word_ends.unsqueeze(1))

        word_lengths = attn_mask.sum(dim=1)
        if torch.any(word_lengths == 0):
            raise ValueError("Found a blank word in training data!  This will break everything, starting with the attention masks, as some rows of the scoring table will be set to entirely -inf and then softmax to NaN.")

        # Only take log where mask is true, else set to -inf directly
        attn_mask_f = torch.full_like(attn_mask, float('-inf'), dtype=torch.float)
        attn_mask_f[attn_mask] = 0.0

        attn_scores = self.attn(bert_out).T  # [1, n_subtokens]
        # Expand without actual new memory
        attn_scores = attn_scores.expand(n_words, n_subtokens)
        attn_scores = attn_mask_f + attn_scores

        return torch.softmax(attn_scores, dim=1)  # [n_words, n_subtokens]

    def _cluster_ids(self, doc: Doc) -> torch.Tensor:
        """
        Args:
            doc: document information

        Returns:
            torch.Tensor of shape [n_word], containing cluster indices for
                each word. Non-coreferent words have cluster id of zero.
        """
        # Precompute lengths and avoid building intermediate dict
        # `doc["cased_words"]` is expected to be a list or tuple, so enumerate is faster here
        n_words = len(doc["cased_words"])
        # Create a flat index tensor filled with zeros
        cluster_ids = torch.zeros(n_words, dtype=torch.long, device=self.device)
        for i, cluster in enumerate(doc["word_clusters"], start=1):
            # cluster is list of indices for this cluster
            cluster_ids.scatter_(0, torch.as_tensor(cluster, device=self.device, dtype=torch.long), i)
        return cluster_ids
