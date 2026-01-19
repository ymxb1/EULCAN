# coding: utf-8
from itertools import groupby
from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F

from utils.utils import gen_length_mask


class CTCPostProcessor(object):


    def __init__(self, vocab: List[str],) -> None:

        self.vocab = vocab
        self._embedding = list(self.vocab) + ['<eos>']

    @staticmethod
    def ctc_best_path(
        logits: torch.Tensor,
        vocab: List[str],
        input_lengths: Optional[torch.Tensor] = None,
        blank: int = 0,
    ) -> List[Tuple[List[str], float]]:

        # compute softmax
        probs = F.softmax(logits.permute(0, 2, 1), dim=1)
        # get char indices along best path
        best_path = torch.argmax(probs, dim=1)  # [N, T]

        if input_lengths is not None:
            length_mask = gen_length_mask(input_lengths, probs.shape).to(
                device=probs.device
            )  # [N, 1, T]
            probs.masked_fill_(length_mask, 1.0)
            best_path.masked_fill_(length_mask.squeeze(1), blank)

        # define word proba as min proba of sequence
        probs, _ = torch.max(probs, dim=1)  # [N, T]
        probs, _ = torch.min(probs, dim=1)  # [N]

        words = []
        for sequence in best_path:
            # collapse best path (using itertools.groupby), map to chars, join char list to string
            collapsed = [vocab[k] for k, _ in groupby(sequence) if k != blank]
            words.append(collapsed)

        return list(zip(words, probs.tolist()))

    def __call__(  # type: ignore[override]
        self, logits: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> List[Tuple[List[str], float]]:

        # Decode CTC
        return self.ctc_best_path(
            logits=logits,
            vocab=self.vocab,
            input_lengths=input_lengths,
            blank=len(self.vocab),
        )
