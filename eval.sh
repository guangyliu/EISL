fair_base=./fairseq
data_bin_dir=${fair_base}/data-bin/wmt14_ende
ckpt=Vanilla/noKD/EISL/checkpoint30000_avg_5.pt

CUDA_VISIBLE_DEVICES=0 python ${fair_base}/fairseq_cli/generate.py \
    ${data_bin_dir} \
--gen-subset test \
--task translation_lev \
--path ${ckpt} \
--iter-decode-max-iter 0 \
--iter-decode-eos-penalty 0 \
--beam 1 --remove-bpe \
--print-step \
--batch-size 512
