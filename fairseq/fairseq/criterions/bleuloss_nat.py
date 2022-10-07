# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from torch.cuda import LongTensor, FloatTensor
import torch
from torch import nn
from collections import Counter
from torch.autograd import Variable
import random
from torch import Tensor
from timeit import default_timer as timer


@dataclass
class BLEULossNatCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")



@register_criterion("ngrambleuloss_drawloss")
class NgramBLEULossNATCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.bestbleu = 0
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, loss_nll = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        
        logging_output = {
            "loss": loss.data,
            "loss_nll": loss_nll.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size
        }

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        #         lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)  # .view(-1)
        loss_nll = F.nll_loss(
            lprobs.view(-1, lprobs.size(-1)),
            target.view(-1),
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        ).detach()
        #         lprobs = lprobs.transpose(0,1)
#         expected_len = expected_length(lprobs)
        loss = sample["ntokens"] * self.batch_log_bleulosscnn_nat(lprobs, target, 4, 1)
        return loss, loss_nll

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_nll_sum = sum(log.get("loss_nll", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "bleu_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss", loss_nll_sum / sample_size / math.log(2), sample_size, round=3
        )
        
        with open('/home/lptang/reuslt.txt','a') as thefile:
            thefile.write(str((loss_nll_sum / sample_size / math.log(2)).item())+'\t'+str((loss_sum / sample_size / math.log(2)).item())+'\n')
        exit()
        
#         if  metrics.BleuLog.bleuce:
#             new_bleu_loss = (loss_nll_sum / sample_size / math.log(2)).item()
#             if new_bleu_loss/metrics.BleuLog.bleu_loss_value1 >= 0.94 and new_bleu_loss/metrics.BleuLog.bleu_loss_value2 >=0.97 and new_bleu_loss < 1.4:
#                 print("Condition achieved")
#                 metrics.BleuLog.bleuce = False
#             metrics.BleuLog.bleu_loss_value1 = metrics.BleuLog.bleu_loss_value2
#             metrics.BleuLog.bleu_loss_value2 = new_bleu_loss
# #         if (loss_nll_sum / sample_size / math.log(2)).item() < 1.2:
            
##########################################
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False

    
    def batch_log_bleulosscnn_nat(self, decoder_outputs, target_idx, ngram_list, trans_len, pad=1,
                              weight_list=None,eos=2):
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
        if metrics.BleuLog.bleuce:
            ngram_list = [tgt_len]
            metrics.BleuLog.tf_ratio = 1.0
        else:
            ngram_list = [2]
#             print(ngram_list)
#             print(ngram_list)
            metrics.BleuLog.tf_ratio = 0.0
#         ngram_list = [1,2,3,4]
#         ngram_list = [1,2,3,4]
#         ngram_list = [2]
#         if type(ngram_list) == int:
#             ngram_list = [ngram_list]
#         if ngram_list[0] <= 0:
#             ngram_list[0] = output_len
        if weight_list is None:
            weight_list = [1. / len(ngram_list)] * len(ngram_list)
#         weight_list = [0.1,0.1,0.4,0.4]
#         weight_list = [0.8,0.2]
        decoder_outputs = torch.relu(decoder_outputs + 20) - 20  # 过滤掉过小的概率  logp = -20 ---> p = 2e-9

        index = target_idx.unsqueeze(1).expand(-1, output_len, tgt_len)

        # [batch, output_len, target_len]
        cost_nll = decoder_outputs.gather(dim=2, index=index)

        # [batch, 1, output_len, target_len] -> [batch, 1, target_len, output_len]
        cost_nll = cost_nll.unsqueeze(1)#.transpose(2, 3) # P(A)log(a)

        out = cost_nll
        sum_gram = FloatTensor([0.])
###########################
        zero = torch.tensor(0.0).cuda()
        target_expand = target_idx.view(batch_size,1,1,-1).expand(-1,-1,output_len,-1)
        out = torch.where(target_expand==pad, zero, out)
        
#         ratio = (target_idx!=pad).sum()*1.0/(target_idx!=0).sum() * tgt_len
#         eos_tensor = target_expand == eos
#         eos_tensor[:,:,-1,:] = False
#         out = torch.where(eos_tensor,zero,out)
###########################
        for cnt, ngram in enumerate(ngram_list):
            if ngram > output_len:
                continue
            eye_filter = torch.eye(ngram).view([1, 1, ngram, ngram]).cuda()
            # out: [batch, 1, output_len, target_len]
            # eye_filter: [1, ngram, ngram]
            # term: [batch, 1, output_len - ngram + 1, target_len - ngram + 1]
############################################
#             ratio = (target_idx!=pad).sum()*1.0/(target_idx!=0).sum() * ngram
            term = nn.functional.conv2d(out, eye_filter)/ngram  # 除以ngram，作为normalization
#             term = nn.functional.conv2d(out, eye_filter)/ratio  # 除以ngram，作为normalization
############################################

            if ngram < decoder_outputs.size()[1]:
                #                 term[torch.isnan(term)] = -float('inf')
                sample_m = 1
                if sample_m == 1:
                    gum_tmp = F.gumbel_softmax(term.squeeze_(1), tau=1, dim=1)
                    #                 term[term==-float('inf')] = 0
                    term = term.mul(gum_tmp).sum(1).mean(1)
                elif sample_m == 2:
                    gum_tmp = F.gumbel_softmax(term.squeeze_(1), tau=1, dim=1).detach()
                    term = term.mul(gum_tmp).sum(1)
                    gum_tmp = F.gumbel_softmax(term, tau=1, dim=1).detach()
                    term = term.mul(gum_tmp).sum(1)
            sum_gram += weight_list[cnt] * term.sum()
        loss = - sum_gram / batch_size
        return loss