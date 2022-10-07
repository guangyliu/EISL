noise=0
for beam in 1 2 3 4
do
((cuda=beam-1))
CUDA_VISIBLE_DEVICES=$cuda fairseq-generate wmt17-bin/rsbs10     --path checkpoints/denoising_bart_wmt/rsbs_bleu/ng_rsbs$noise'_ba_0.0_12_'$noise/checkpoint_best.pt     --batch-size 256 --beam $beam --remove-bpe --gen-subset valid --quiet &
sleep 2
done
 
#  --path checkpoints/denoising_bart_wmt/rsbs/cr_rsbs$noise'_ba_1.0_cetf_wmt_'$noise/checkpoint_best.pt  