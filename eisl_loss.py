"""Edit-Invariant Sequence Loss (EISL) as a stand-alone PyTorch function.

Liu et al., "Don't Take It Literally: An Edit-Invariant Sequence Loss for Text Generation",
NAACL 2022. https://aclanthology.org/2022.naacl-main.150/

The loss slides every reference n-gram over every position of the model's output
distribution (a 2-D convolution with an identity kernel on the [output_len x target_len]
log-probability matrix), soft-selects the best position with a Gumbel softmax, and averages
over reference n-grams. With n == target length it reduces to the usual sequence
cross-entropy, so it is a drop-in replacement for CE.

Only depends on torch. Framework-specific wrappers: `EISL.py` (fairseq criterion) and
`EISL_trainer.py` (Hugging Face Trainer).
"""
from typing import Iterable, Optional, Sequence, Union

import torch
import torch.nn.functional as F


def _resolve_ngrams(ngram: Union[int, Iterable[int]], length: int) -> list:
    """Positive values are n-gram orders; non-positive values are offsets from `length`
    (0 -> full length = cross-entropy, -1 -> length-1, ...)."""
    if isinstance(ngram, int):
        ngram = [ngram]
    out = set()
    for n in ngram:
        n = n if n > 0 else length + n
        if 0 < n <= length:
            out.add(n)
    return sorted(out) or [length]


def eisl_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    ngram: Union[int, Sequence[int]] = (1, 2, 3, 4),
    pad_id: Optional[int] = None,
    weights: Optional[Sequence[float]] = None,
    tau: float = 1.0,
    clamp_min: float = -20.0,
) -> torch.Tensor:
    """
    log_probs : [B, T_out, V]  log-softmax outputs of the model (any decoding order).
    targets   : [B, T_tgt]     reference token ids (T_tgt may differ from T_out).
    ngram     : n-gram orders to match, e.g. (1, 2, 3, 4); 0 means "full length" (== CE).
    pad_id    : reference n-grams that contain `pad_id` are ignored.
    weights   : per-order weights, default uniform.
    tau       : Gumbel-softmax temperature for position selection.
    clamp_min : log-probs below this are clamped (numerical safety, as in the paper code).

    Returns a scalar loss (mean over batch). Minimise it.
    """
    B, T_out, V = log_probs.shape
    T_tgt = targets.shape[1]
    orders = _resolve_ngrams(ngram, min(T_out, T_tgt))
    if weights is None:
        weights = [1.0 / len(orders)] * len(orders)
    assert len(weights) == len(orders)

    lp = torch.clamp(log_probs, min=clamp_min)
    # log p(y_j = y*_k) for every output position j and reference position k -> [B, 1, T_out, T_tgt]
    cost = lp.gather(2, targets.unsqueeze(1).expand(B, T_out, T_tgt)).unsqueeze(1)
    valid = None
    if pad_id is not None:
        valid = (targets != pad_id).float()  # [B, T_tgt]

    total = log_probs.new_zeros(())
    for w, n in zip(weights, orders):
        eye = torch.eye(n, dtype=cost.dtype, device=cost.device).view(1, 1, n, n)
        # mean log-prob of the reference n-gram starting at k placed at output position j
        term = F.conv2d(cost, eye).squeeze(1) / n           # [B, T_out-n+1, T_tgt-n+1]
        sel = F.gumbel_softmax(term, tau=tau, dim=1)         # soft position selection
        per_ngram = (term * sel).sum(1)                      # [B, T_tgt-n+1]
        if valid is not None:
            # an n-gram is valid only if none of its tokens is padding
            m = F.avg_pool1d(valid.unsqueeze(1), kernel_size=n, stride=1).squeeze(1)
            m = (m > 0.999).float()
            per_ngram = (per_ngram * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:
            per_ngram = per_ngram.mean(1)
        total = total + w * per_ngram.sum()
    return -total / B


__all__ = ["eisl_loss"]
