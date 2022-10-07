# EISL
Source code of paper: Don’t Take It Literally: An Edit-Invariant Sequence Loss for Text Generation

https://arxiv.org/abs/2106.15078

NAACL 2022 Main Conference, Oral Presentation
 
## Usage (Fairseq)
Put the *EISL.py* file in to *fairseq/fairseq/criterions/EISL.py*, then you can train with EISL loss by adding *--criterion EISL* into fairseq command.

Please refer to [fairseq](./fairseq) for more implementation details.
### Non-Autoregressive Machine Translation
For the NAT experiments, NAT codes (fairseq) and models are released in the NAT folder. We also provide the models trained by CE loss. The results are reported in the paper.

## Usage (HuggingFace, Need to be verified)
You can override the *compute_loss* function of Trainer like shown in *EISL_trainer.py*. 