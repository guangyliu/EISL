# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion,LegacyFairseqCriterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from torch.cuda import LongTensor, FloatTensor
import torch
from torch import nn
from collections import Counter
from torch.autograd import Variable
import random

@dataclass
class BLEULossCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

@register_criterion("ngrambleuloss_nat")
class NgramBLEULossNATCriterion(LegacyFairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args,task)
        self.sentence_avg = False
        self.bestbleu = 0
        self.ngram = [int(i) for i in args.ngram.split(',')]

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--ngram', default='1,2', type=str)
        # fmt: on
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
#         loss_nll = F.nll_loss(
#             lprobs.view(-1, lprobs.size(-1)),
#             target.view(-1),
#             ignore_index=self.padding_idx,
#             reduction="sum" if reduce else "none",
#         ).detach()
        #         lprobs = lprobs.transpose(0,1)
        expected_len = expected_length(lprobs)
        loss = sample["ntokens"] * self.batch_log_bleulosscnn_nat(lprobs, target, 4, expected_len)
        return loss, loss

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
        
#         with open('/home/lptang/reuslt.txt','a') as thefile:
#             thefile.write(str((loss_nll_sum / sample_size / math.log(2)).item())+'\t'+str((loss_sum / sample_size / math.log(2)).item())+'\n')
#         exit()
        
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
#         if metrics.BleuLog.bleuce:
#             ngram_list = [tgt_len]
#             metrics.BleuLog.tf_ratio = 1.0
#         else:
#             ngram_list = self.ngram
#             metrics.BleuLog.tf_ratio = 0.0
        ngram_list = self.ngram
        if ngram_list[0] == -1:
            ngram_list = [tgt_len]
        
        if len(ngram_list)==2:
            if ngram_list[0] == 1 and ngram_list[1] == 2:
                weight_list = [0.8,0.2]
        if weight_list is None:
            weight_list = [1. / len(ngram_list)] * len(ngram_list)
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
    

@register_criterion("ngrambleuloss_nat1234")
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
        expected_len = expected_length(lprobs)
        loss = sample["ntokens"] * self.batch_log_bleulosscnn_nat(lprobs, target, 4, expected_len)
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
        
        if  metrics.BleuLog.bleuce:
            new_bleu_loss = (loss_nll_sum / sample_size / math.log(2)).item()
            if new_bleu_loss/metrics.BleuLog.bleu_loss_value1 >= 0.94 and new_bleu_loss/metrics.BleuLog.bleu_loss_value2 >=0.97 and new_bleu_loss < 1.4:
                print("Condition achieved")
                metrics.BleuLog.bleuce = False
            metrics.BleuLog.bleu_loss_value1 = metrics.BleuLog.bleu_loss_value2
            metrics.BleuLog.bleu_loss_value2 = new_bleu_loss
#         if (loss_nll_sum / sample_size / math.log(2)).item() < 1.2:
            
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
            ngram_list = [1,2,3,4]
            metrics.BleuLog.tf_ratio = 1.0
        else:
            ngram_list = [tgt_len]
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
#         weight_list = [0.3,0.3,0.3,0.1]
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
    



@register_criterion("ngrambleuloss")
class NgramBLEULossCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.cnt = 0
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
        expected_len = expected_length(lprobs)
        loss = sample["ntokens"] * self.batch_log_bleulosscnn(lprobs, target, 4, expected_len)
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
    def batch_log_bleulosscnn(self, decoder_outputs, target_variable, ngram_list, translation_len, pad=1, weight_list=None):
        """
        decoder_outputs - matrix with probabilityes  -- log domain  Batch x Length x Vocab
        target_variable - reference batch   Batch x Length
        maxorder - max order of n-gram   
        translation_lengths -  lengths of the translations - torch tensor    Batch
        reference_lengths - lengths of the references - torch tensor
        """
#         import ipdb
#         ipdb.set_trace()
#         ngram_list = [decoder_outputs.size()[-2]- metrics.BleuLog.per_gram*metrics.BleuLog.minus_gram]
        ngram_list = [decoder_outputs.size()[-2]]
        if ngram_list[0] <= 0:
            ngram_list[0] = decoder_outputs.size()[-2]
#         ngram_list = [decoder_outputs.size()[-2]-3]
        if type(ngram_list) == int:
            ngram_list = [ngram_list]
        if weight_list is None:
            weight_list = [1./len(ngram_list)] * len(ngram_list)
        n_words = decoder_outputs.size()[-1]  # Vocab_size
        batch_size = decoder_outputs.size()[0] # Batch size
        decoder_outputs = torch.relu(decoder_outputs + 20) - 20 # 过滤掉过小的概率  logp = -20 ---> p = 2e-9
        reference_lengths = (target_variable != pad).sum(-1) # Batch (每个句子除去pad的实际长度)
        target_variable = target_variable.contiguous().view(-1, 1) # (batch*length) x 1 串在一起
        target_length = target_variable.size()[0] # 总长度
        pred_onehot = torch.cat(decoder_outputs.chunk(batch_size, 0), -1).transpose(1, 2).unsqueeze(-2)
        # chunk 后有 batch 个 [1 x length x vocab]
        target_onehot = torch.zeros(target_length, n_words).cuda().scatter_(1, target_variable, 1).view(
            (target_length, -1, 1, 1))
        target_onehot[:, pad, 0, 0] = 0  # pad
        out = nn.functional.conv2d(pred_onehot, target_onehot, groups=batch_size)  # O^T  
        out = torch.cat(out.chunk(batch_size, 1), 0).permute(0, 2, 3, 1)  # O
        sum_gram = FloatTensor([0.])

        for cnt, ngram in enumerate(ngram_list):
            eye_filter = torch.eye(ngram).view([1, 1, ngram, ngram]).cuda()
#             if ngram < decoder_outputs.size()[1]:
#                 out[out==0] = float('nan')
            term = nn.functional.conv2d(out, eye_filter)
            if ngram < decoder_outputs.size()[1]:
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
                elif sample_m == 3: # topk  n-2 也就3个选
                    term += torch.randn_like(term)*decoder_outputs.size()[-2]/5
                    term = term.max(1)[0].sum() 
                elif sample_m == 4:
                    term += torch.randn_like(term)
                    term = term.max(1)[0]
                    gum_tmp = F.gumbel_softmax(term,tau=1,dim=1).detach()
                    term = term.mul(gum_tmp).sum()
                elif sample_m == 5:
                    term += torch.randn_like(term)
                    term = term.max(1)[0]
                    gum_tmp = F.gumbel_softmax(term,tau=1,dim=1,hard=True).detach()
                    term = term.mul(gum_tmp).sum()
            sum_gram += weight_list[cnt]*term.sum()
        loss = - sum_gram / batch_size
        return loss
    

    def batch_log_bleulosscnn_bk(self, decoder_outputs, target_variable, ngram_list, translation_len, pad=1, weight_list=None):
        """
        decoder_outputs - matrix with probabilityes  -- log domain
        target_variable - reference batch
        maxorder - max order of n-gram
        translation_lengths -  lengths of the translations - torch tensor
        reference_lengths - lengths of the references - torch tensor
        """
#         ngram_list = [decoder_outputs.size()[-2], decoder_outputs.size()[-2]-1, decoder_outputs.size()[-2]-2]
        ngram_list = [decoder_outputs.size()[-2]-1]
        # 如果 ngram小于1,则没有意义, 直接跳过
        if type(ngram_list) == int:
            ngram_list = [ngram_list]
        if weight_list is None:
            weight_list = [1./len(ngram_list)] * len(ngram_list)
        n_words = decoder_outputs.size()[-1]
        batch_size = decoder_outputs.size()[0]
        decoder_outputs = torch.relu(decoder_outputs + 20) - 20 # 过滤掉过小的概率  logp = -20 ---> p = 2e-9
        reference_lengths = (target_variable != pad).sum(-1)
        target_variable = target_variable.contiguous().view(-1, 1)
        target_length = target_variable.size()[0]
        pred_onehot = torch.cat(decoder_outputs.chunk(batch_size, 0), -1).transpose(1, 2).unsqueeze(-2)
        target_onehot = torch.zeros(target_length, n_words).cuda().scatter_(1, target_variable, 1).view(
            (target_length, -1, 1, 1))
        target_onehot[:, pad, 0, 0] = 0  # pad
        out = nn.functional.conv2d(pred_onehot, target_onehot, groups=batch_size)
        out = torch.cat(out.chunk(batch_size, 1), 0).permute(0, 2, 3, 1)
        sum_gram = FloatTensor([0.])
        import ipdb
        ipdb.set_trace()
        for cnt, ngram in enumerate(ngram_list):
            eye_filter = torch.eye(ngram).view([1, 1, ngram, ngram]).cuda()
            if ngram < decoder_outputs.size()[1]:
#                 out[out==0] = -float('inf')
                term = nn.functional.conv2d(out, eye_filter)
#                 gum_tmp = F.gumbel_softmax(term.squeeze_(1), tau=1, dim=1)
#                 term[term==-float('inf')] = 0
#                 term[torch.isnan(term)] = -20*ngram
                gum_tmp = F.gumbel_softmax(term.squeeze_(1), tau=1, dim=2)
#                 term = term.mul(gum_tmp).sum(1).mean(1)
                term = term.mul(gum_tmp).mean(1).sum(1)
            else:
#                 out[out==-float('inf')] = 0
                term = nn.functional.conv2d(out, eye_filter)
            sum_gram += weight_list[cnt]*term.sum()
        loss = - sum_gram / batch_size
        return loss
    
    def batch_log_bleulosscnn_nat(self, decoder_outputs, target_idx, ngram_list, pad=1,
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

        ################################# NEW #########################################3
        # start = timer()

        # [batch_size, output_len, target_len]
        index = target_idx.unsqueeze(1).expand(-1, output_len, tgt_len)

        # [batch, output_len, target_len]
        cost_nll = decoder_outputs.gather(dim=2, index=index)

        # [batch, 1, output_len, target_len] -> [batch, 1, target_len, output_len]
        cost_nll = cost_nll.unsqueeze(1)#.transpose(2, 3)

        # gather_time_used = timer() - start

        ################################# NEW DONE #########################################3

        ################################# OLD #########################################3
        # start = timer()

        # # [batch * tgt_len, 1]
        # target_variable = target_idx.contiguous().view(-1, 1)
        # target_token_count = target_variable.size()[0]
        #
        # # [1, batch * vocab, 1, sentence_len]
        # pred_onehot = torch.cat(decoder_outputs.chunk(batch_size, 0), -1).transpose(1, 2).unsqueeze(-2)
        #
        # # [batch * sentence_len, vocab, 1, 1]
        # target_onehot = torch.zeros(target_token_count, vocab_size).cuda().scatter_(1, target_variable, 1).view(
        #     (target_token_count, -1, 1, 1))
        # # target_onehot[:, pad, 0, 0] = 0  # pad
        #
        # # print('batch_size',batch_size)
        # # assert pred_onehot.size(1) % batch_size == 0
        #
        # # [1, batch * sentence_len, 1, sentence_len]
        # out = nn.functional.conv2d(pred_onehot, target_onehot, groups=batch_size)
        #
        # # [batch, 1, sentence_len, sentence_len]
        # out = torch.cat(out.chunk(batch_size, 1), 0).permute(0, 2, 3, 1)

        # conv_time_used = timer() - start

        ################################# OLD DONE #########################################3

        # speedup = conv_time_used / gather_time_used
        # print("batch_size: {},".format(batch_size),
        #       "gather time: {:.5f},".format(gather_time_used),
        #       "conv time: {:.5f},".format(conv_time_used),
        #       "speed up: {:.2f}x".format(speedup))

        # # "cost_nll" should equal to "out"
        # if batch_size > 1:
        #     assert (cost_nll - out).abs().sum() < 1e-5
        # else:
        #     # when batch_size == 1, it's strange the result is transposed
        #     assert (cost_nll.transpose(2, 3) - out).abs().sum() < 1e-5

        out = cost_nll

        sum_gram = FloatTensor([0.])

        for cnt, ngram in enumerate(ngram_list):
            eye_filter = torch.eye(ngram).view([1, 1, ngram, ngram]).cuda()

            # out: [batch, 1, output_len, target_len]
            # eye_filter: [1, ngram, ngram]
            # term: [batch, 1, output_len - ngram + 1, target_len - ngram + 1]
            term = nn.functional.conv2d(out, eye_filter)

            if ngram < decoder_outputs.size()[1]:
                #                 term[torch.isnan(term)] = -float('inf')
                sample_m = 2
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
    
    

@register_criterion("bleulossexp")
class BLEULossexpCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

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
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        #         lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)  # .view(-1)
        # loss_nll = F.nll_loss(
        #     lprobs.view(-1, lprobs.size(-1)),
        #     target.view(-1),
        #     ignore_index=self.padding_idx,
        #     reduction="sum" if reduce else "none",
        # ).detach()
        #         lprobs = lprobs.transpose(0,1)
        expected_len = expected_length(lprobs)
        loss = sample["ntokens"] * self.batch_bleulosscnn(lprobs, target, 4, expected_len)
        return loss, loss

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


    def batch_bleulosscnn(self, decoder_outputs, target_variable, maxorder, translation_len, pad=1):
        """
        decoder_outputs - matrix with probabilityes -- exp domain
        r - reference batch
        maxorder - max order of n-gram
        translation_lengths -  lengths of the translations - torch tensor
        reference_lengths - lengths of the references - torch tensor
        """
        weights = [0.1, 0.3, 0.3, 0.3]
        decoder_outputs = torch.exp(decoder_outputs)
        n_words = decoder_outputs.size()[-1]
        batch_size = decoder_outputs.size()[0]
        decoder_outputs = torch.relu(decoder_outputs - 2E-9) + 2E-9  # 较小的概率直接按0处理, 实际应用时可能要调整
        reference_lengths = (target_variable != pad).sum(-1)
        r = target_variable.tolist()
        target_variable = target_variable.contiguous().view(-1, 1)
        target_length = target_variable.size()[0]
        pred_onehot = torch.cat(decoder_outputs.chunk(batch_size, 0), -1).transpose(1, 2).unsqueeze(-2)
        target_onehot = torch.zeros(target_length, n_words).cuda().scatter_(1, target_variable, 1).view(
            (target_length, -1, 1, 1))
        target_onehot[:, pad, 0, 0] = 0  # pad
        out = nn.functional.conv2d(pred_onehot, target_onehot, groups=batch_size)
        out = torch.cat(out.chunk(batch_size, 1), 0)

        r_cnt = [ngram_ref_counts(r, reference_lengths.tolist(), j + 1) for j in range(maxorder)]
        alist = [out.permute(0, 2, 3, 1)]  ##### double !!!
        gram = [None] * maxorder
        for j in range(1, maxorder):
            if j < target_length:
                alist.append(torch.mul(alist[j - 1][:, :, :-1, :-1], alist[0][:, :, j:, j:]))
                for ii in range(batch_size):
                    alist[j][ii, :, :, reference_lengths[ii] - j:] = 0
            else:
                break
        sum_gram = FloatTensor([0.])
        for j in range(maxorder):
            if j <= target_length:
                alist[j] = torch.pow(alist[j], 1.0 / (j + 1))
                An_tmp = -alist[j] + torch.sum(alist[j], 2, keepdim=True) + 1
                second_arg = torch.cat([r_cnt[j][[i]] / An_tmp[[i], 0] for i in range(batch_size)], 0).unsqueeze(1)
                term = torch.min(alist[j], alist[j] * second_arg)
                #             term = alist[j]
                gram[j] = term.sum([1, 2, 3]) / (torch.relu(reference_lengths - j) + 1e-2)  # size: batch_size
                sum_gram += weights[j] * torch.log(gram[j]).sum()  # 1.0/ maxorder改为 weights list
            else:
                break
        bp_tmp = torch.true_divide(translation_len, reference_lengths)
        bp_tmp = F.hardtanh(1 - 1 / bp_tmp, -9, 0)
        # for ii in range(batch_size):
        #     bp_tmp[ii] = 0 if bp_tmp[ii].item() > 1 else -4.6 if bp_tmp[ii].item() < 1e-1 else 1 - 1. / bp_tmp[ii]
        loss = -(bp_tmp.sum() + sum_gram) / batch_size
        # import ipdb
        # ipdb.set_trace()
        return loss

@register_criterion("bleuloss")
class BLEULossCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.cnt = 0
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
        expected_len = expected_length(lprobs)
        loss = sample["ntokens"] * self.batch_log_bleulosscnn(lprobs, target, 4, expected_len)
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

    def batch_log_bleulosscnn_bk0123(self, decoder_outputs, target_variable, maxorder, translation_len, pad=1):
        """
        decoder_outputs - matrix with probabilityes  -- log domain
        target_variable - reference batch
        maxorder - max order of n-gram
        translation_lengths -  lengths of the translations - torch tensor
        reference_lengths - lengths of the references - torch tensor
        """

        weights = [0.1, 0.3, 0.3, 0.3]
        n_words = decoder_outputs.size()[-1]
        batch_size = decoder_outputs.size()[0]
        #     decoder_outputs = torch.log(decoder_outputs)
        #     decoder_outputs = torch.relu(decoder_outputs + 20) - 20
        #         target_variable = LongTensor(r).view(-1, 1).cuda()  #
        reference_lengths = (target_variable != pad).sum(-1)
        target_variable = target_variable.contiguous().view(-1, 1)

        target_length = target_variable.size()[0]
        pred_onehot = torch.cat(decoder_outputs.chunk(batch_size, 0), -1).transpose(1, 2).unsqueeze(-2)
        target_onehot = torch.zeros(target_length, n_words).cuda().scatter_(1, target_variable, 1).view(
            (target_length, -1, 1, 1))
        target_onehot[:, pad, 0, 0] = 0  # pad
        out = nn.functional.conv2d(pred_onehot, target_onehot, groups=batch_size)
        out = torch.cat(out.chunk(batch_size, 1), 0)
        # r_cnt = [ngram_ref_counts(r, reference_lengths.tolist(), j + 1) for j in range(maxorder)]
        alist = [out.permute(0, 2, 3, 1)]
        gram = [None] * maxorder
        alist[0][alist[0] == 0] = 1e3
        for j in range(1, maxorder):
            if j < target_length:
                alist.append((alist[j - 1][:, :, :-1, :-1] * j + alist[0][:, :, j:, j:]) / (j + 1))
                alist[j][alist[j] > 0] = 0
#                 for ii in range(batch_size):
#                     alist[j][ii, :, :, reference_lengths[ii] - j:] = 0
        alist[0][alist[0] > 0] = 0
        sum_gram = FloatTensor([0.])
        for j in range(maxorder):
            gum_tmp = F.gumbel_softmax(alist[j].squeeze(1), tau=1, dim=1)
            term = alist[j].squeeze(1).mul(gum_tmp).sum(1).mean(1)
            gram[j] = term  # /  (torch.relu(reference_lengths - j) + 1e-2)
            #         sum_gram += 1.0 / maxorder * gram[j].sum()  # 1.0/ maxorder改为 weights list
            sum_gram += weights[j] * gram[j].sum()
        bp_tmp = torch.true_divide(translation_len, reference_lengths)
        bp_tmp = F.hardtanh(1 - 1 / bp_tmp, -9, 0)
        loss = -(bp_tmp.sum() + sum_gram) / batch_size
        return loss

    def batch_log_bleulosscnn(self, decoder_outputs, target_variable, maxorder, translation_len, pad=1):
        """
        decoder_outputs - matrix with probabilityes  -- log domain
        target_variable - reference batch
        maxorder - max order of n-gram
        translation_lengths -  lengths of the translations - torch tensor
        reference_lengths - lengths of the references - torch tensor
        """
        self.cnt += 1
        # if self.cnt > 620:
        #     import ipdb
        #     ipdb.set_trace()
        weights = [0.1, 0.3, 0.3, 0.3]
        n_words = decoder_outputs.size()[-1]
        batch_size = decoder_outputs.size()[0]
        decoder_outputs = torch.relu(decoder_outputs + 20) - 20 # 过滤掉过小的概率  logp = -20 ---> p = 2e-9
        reference_lengths = (target_variable != pad).sum(-1)
        target_variable = target_variable.contiguous().view(-1, 1)

        target_length = target_variable.size()[0]
        pred_onehot = torch.cat(decoder_outputs.chunk(batch_size, 0), -1).transpose(1, 2).unsqueeze(-2)
        target_onehot = torch.zeros(target_length, n_words).cuda().scatter_(1, target_variable, 1).view(
            (target_length, -1, 1, 1))
        target_onehot[:, pad, 0, 0] = 0  # pad
        out = nn.functional.conv2d(pred_onehot, target_onehot, groups=batch_size)
        out = torch.cat(out.chunk(batch_size, 1), 0).permute(0, 2, 3, 1)
        out_ori = torch.clone(out)
        out_ori[out_ori == 0] = float('inf')
        sum_gram = FloatTensor([0.])
        for j in range(maxorder):
            if 0 < j < target_length:
                out = (out[:, :, :-1, :-1]*j + out_ori[:, :, j:, j:]) / (j + 1)
                out[out == float('inf')] = 0
            gum_tmp = F.gumbel_softmax(out.squeeze(1), tau=1, dim=1)
#             term = out.squeeze(1).mul(gum_tmp).sum(1).mean(1) 最后一个mean 要考虑pad , 所以改为下面计算
            term = out.squeeze(1).mul(gum_tmp).sum(1).sum(1)
            term = torch.true_divide(term, torch.relu(reference_lengths-j-1)+1)
            sum_gram += weights[j] * term.sum()
        bp_tmp = torch.true_divide(translation_len, reference_lengths)
        bp_tmp = F.hardtanh(1 - 1 / bp_tmp, -9, 0)
        loss = -(bp_tmp.sum() + sum_gram) / batch_size
        return loss

def expected_length(decoder_outputs, eos_id=2):
    pred_length = decoder_outputs.size()[1]
    tmp_tensor = FloatTensor([idd + 1 for idd in range(pred_length)])
    return F.softmax(decoder_outputs[:, :, eos_id], -1).matmul(tmp_tensor)


def argmax_length(decoder_outputs, eos_id=2):
    (decoder_outputs.max(-1)[1] == eos_id).max(1)[1]
    pred_length = decoder_outputs.size()[1]
    tmp_tensor = FloatTensor([idd + 1 for idd in range(pred_length)])
    return F.softmax(decoder_outputs[:, :, eos_id], -1).matmul(tmp_tensor)


def ngram_ref_counts(reference, lengths, n):
    """
    For each position counts n-grams equal to n-gram to this position
    reference - matrix sequences of id's from vocabulary.[batch, ref len]
    NOTE reference should be padded with some special ids
    At least one value in length must be equal reference.shape[1]
    output: counts n-grams for each start position padded with zeros
    """
    res = []
    max_len = max(lengths)
    if max_len - n + 1 <= 0:
        return None
    for r, l in zip(reference, lengths):
        picked = set()  # we only take into accound first appearence of n-gram
        #             (which contains it's count of occurrence)
        current_length = l - n + 1
        cnt = Counter([tuple([r[i + j] for j in range(n)]) \
                       for i in range(current_length)])
        occurrence = []
        for i in range(current_length):
            n_gram = tuple([r[i + j] for j in range(n)])
            val = 0
            if not n_gram in picked:
                val = cnt[n_gram]
                picked.add(n_gram)
            occurrence.append(val)
        padding = [0 for _ in range(max_len - l if current_length > 0 \
                                        else max_len - n + 1)]
        res.append(occurrence + padding)
    return Variable(FloatTensor(res), requires_grad=False)
