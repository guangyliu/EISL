#!/bin/sh
cuda=2
batch_size="64"
# data: iwslt14.tokenized.de-en    multi30k_de_en wmt17_en_de
data="wmt17_en_de"
# loss: cross_entropy  bleuloss cebleuloss ngrambleuloss
loss="ngrambleuloss"
# model: transformer_iwslt_de_en   lstm_wiseman_iwslt_de_en_gumbel
model="lstm_wiseman_iwslt_de_en_gumbel"
# restore-file if use pretrained model
pre=0

# sample 
sample="greedy"
# lr
lr="1e-3"
tf_ratio="1.0"
max_epoch=30
sym="bleu_ce_2gpu_b64"
preix="wmt_ende_ce"
lr_sch="fixed"
# warmup=1000
restore_file="checkpoints/ngram/ng_iw_ls_greedy_ngram_ce_test/checkpoint_best.pt"

if [ ! -d "log/out/"$preix ]; then
  mkdir "log/out/"$preix
  echo "create folder log/out/"$preix
fi

if [ $pre == 1 ]
then 
    name="pre_"${loss:0:2}"_"${data:0:2}"_"${model:0:2}"_"$sample"_"$sym
    CUDA_VISIBLE_DEVICES=$cuda  nohup /home/lptang/anaconda3/envs/torch/bin/fairseq-train    data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr $lr --lr-scheduler $lr_sch   --weight-decay 0.0001     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses     --eval-bleu-remove-bpe   --eval-bleu-print-samples  --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
    --log-interval 20 --tensorboard-logdir log/tf/$preix/$name --restore-file $restore_file --reset-optimizer --sample-method $sample --tf-ratio $tf_ratio \
    --save-interval 1  --clip-norm 0.1 --max-epoch $max_epoch > log/out/$preix/$name.out 2>&1 &
    echo "pre, "$sample", tf: "$tf_ratio", model: "${model:0:4}", "${data:0:7}", "$loss", "$name;
else
    name=${loss:0:2}"_"${data:0:2}"_"${model:0:2}"_"$sample"_"$sym
    CUDA_VISIBLE_DEVICES=$cuda,3 nohup /home/lptang/anaconda3/envs/torch/bin/fairseq-train     data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr $lr --lr-scheduler $lr_sch  --weight-decay 0.0001     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses     --eval-bleu-remove-bpe   --eval-bleu-print-samples  --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
    --log-interval 20 --tensorboard-logdir log/tf/$preix/$name --sample-method $sample --tf-ratio $tf_ratio \
    --save-interval 5 --clip-norm 0.1 --max-epoch $max_epoch > log/out/$preix/$name.out 2>&1 &
    echo "not pre, " $sample", tf: "$tf_ratio", model: "${model:0:4}", "${data:0:7}", "$loss", "$name;
fi

while :
do
    pid=$(ps -ef --sort=start_time | grep fairseq-train | grep -v grep | grep "99" | grep "00:00:" | awk '{print $2}')
    if [ $pid ]
    then
        break
    fi
    sleep 0.5
done
echo "pid is $pid"
echo -e "$batch_size $data $loss $model $sample $lr $tf_ratio $max_epoch $lr_sch $warmup.\nTest the results of different begin checkpoint, 10, 20, 30, best from $restore_file, and tf ratio=0">>log/out/$preix/README.txt
echo -e "$pid $name\n\n">>log/out/$preix/README.txt
