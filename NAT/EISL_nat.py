import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch
from torch import nn

from torch import Tensor


@dataclass
class EISLNatCrfCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")



@register_criterion("EISL_nat")
class EISLNatCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing, ngram, ce_factor, ngram_factor):
        super().__init__(task)
        self.label_smoothing = label_smoothing
        self.ce_factor = ce_factor
        self.ngram_factor = ngram_factor

        self.ngram = [int(n) for n in ngram.split(',')]


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )
        parser.add_argument(
            "--ngram",
            default=None,
            type=str,
            help="the ngram to consider, comma separated, e.g. \"--ngram 2,3,4,-1\" (0 means output_length, -1 means output_length-1)",
        )
        parser.add_argument(
            "--ce-factor",
            required=True,
            type=float,
            help="blend factor for cross entropy",
        )
        parser.add_argument(
            "--ngram-factor",
            required=True,
            type=float,
            help="blend factor for ngram loss",
        )


    def _compute_loss(
            self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                        nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        losses, nll_loss = [], []
        ngram_loss = None
        ce_loss = None
        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                if obj == 'word_ins':
                    _losses = self.compute_EISL(
                        outputs[obj].get("out"),
                        outputs[obj].get("tgt"),
                        outputs[obj].get("mask", None),
                        outputs[obj].get("ls", 0.0),
                        name=obj + "-loss",
                        factor=outputs[obj].get("factor", 1.0),
                    )
                    ngram_loss = _losses.get("ngram_loss")
                    ce_loss = _losses.get("ce_loss")
                else:
                    _losses = self._compute_loss(
                        outputs[obj].get("out"),
                        outputs[obj].get("tgt"),
                        outputs[obj].get("mask", None),
                        outputs[obj].get("ls", 0.0),
                        name=obj + "-loss",
                        factor=outputs[obj].get("factor", 1.0),
                    )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ngram_loss": ngram_loss.data,
            "ce_loss": ce_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "ce_factor": self.ce_factor,
            "ngram_factor":self.ngram_factor
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def config_ngram_list(self, output_length):
        ngram_list = set()
        for n in self.ngram:
            if n>0:
                if n<=output_length:
                    ngram_list.add(n)
            else:
                real_n = output_length+n
                if 0 <real_n:
                    ngram_list.add(real_n)
        if ngram_list:
            ngram_list = list(ngram_list)
        else:
            ngram_list = [output_length]

        return ngram_list

    def compute_EISL(self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0):
        ce_loss = self._compute_loss(
            outputs=outputs,
            targets=targets,
            masks=masks,
            label_smoothing=label_smoothing,
            name=name,
            factor=factor
        )

        log_probs = F.log_softmax(outputs, dim=-1)

        ngram_list = self.config_ngram_list(output_length=outputs.size(1))
        ngram_loss = self.batch_log_EISL_cnn(log_probs, targets, ngram_list=ngram_list)

        eisl_loss = ngram_loss * self.ngram_factor + ce_loss['loss'] * self.ce_factor

        return {"name": 'EISL-loss', "loss": eisl_loss,
                "ngram_loss": ngram_loss,
                "ce_loss": ce_loss['loss'],
                "nll_loss": ce_loss['nll_loss'],
                "factor": 1.0}

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get("loss", 0) for log in logging_outputs)
        ce_loss = sum(log.get("ce_loss", 0) for log in logging_outputs)
        nll_loss = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ngram_loss = sum(log.get("ngram_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        ce_factor = logging_outputs[0].get('ce_factor')
        ngram_factor = logging_outputs[0].get('ngram_factor')

        # we divide by log(2) to convert the loss from base e to base 2

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ce_loss", ce_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ngram_loss", ngram_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ce_factor", ce_factor, sample_size, round=3
        )
        metrics.log_scalar(
            "ngram_factor", ngram_factor, sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False

    def batch_log_EISL_cnn(self, decoder_outputs, target_idx, ngram_list, pad=1,
                              weight_list=None):
        """
        decoder_outputs: [batch_size, output_len, vocab_size]
            - matrix with probabilityes  -- log probs
        target_variable: [batch_size, target_len]
            - reference batch
        ngram_list: int or List[int]
            - n-gram to consider
        pad: int
            the idx of "pad" token
        weight_list : List
            corresponding weight of ngram

        NOTE: output_len == target_len
        """

        batch_size, output_len, vocab_size = decoder_outputs.size()
        _, tgt_len = target_idx.size()

        if type(ngram_list) == int:
            ngram_list = [ngram_list]
        if ngram_list[0] <= 0:
            ngram_list[0] = output_len
        if weight_list is None:
            weight_list = [1. / len(ngram_list)] * len(ngram_list)

        decoder_outputs = torch.relu(decoder_outputs + 20) - 20  # 过滤掉过小的概率  logp = -20 ---> p = 2e-9

        # [batch_size, output_len, target_len]
        index = target_idx.unsqueeze(1).expand(-1, output_len, tgt_len)

        # [batch, output_len, target_len]
        cost_nll = decoder_outputs.gather(dim=2, index=index)

        # [batch, 1, output_len, target_len]
        cost_nll = cost_nll.unsqueeze(1)

        sum_gram = torch.tensor([0.], dtype=cost_nll.dtype, device=cost_nll.device)

        for cnt, ngram in enumerate(ngram_list):
            # out: [batch, 1, output_len, target_len]
            # eye_filter: [1, 1, ngram, ngram]
            eye_filter = torch.eye(ngram).view([1, 1, ngram, ngram]).cuda()

            assert ngram <= decoder_outputs.size()[1]
            # term: [batch, 1, output_len - ngram + 1, target_len - ngram + 1]
            term = nn.functional.conv2d(cost_nll, eye_filter) / ngram

            # maybe dim should be 2, but sometime 1 is better
            gum_tmp = F.gumbel_softmax(term.squeeze_(1), tau=1, dim=1)

            term = term.mul(gum_tmp).sum(1).mean(1)

            sum_gram += weight_list[cnt] * term.sum()
        loss = - sum_gram / batch_size
        return loss



