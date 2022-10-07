#!/bin/sh
cuda=2
batch_size="256"
# data: iwslt14.tokenized.de-en    multi30k_de_en
data="wmt16_en_de_bpe32k"
# loss: cross_entropy  bleuloss cebleuloss ngrambleuloss ngrambleulossNAT
loss="cross_entropy"
# model: transformer_iwslt_de_en   lstm_wiseman_iwslt_de_en_gumbel  
model="transformer_vaswani_wmt_en_de_big"
# restore-file if use pretrained model
pre=1

# sample 
sample="topk"
top_k=45
# lr
lr="5e-4"
tf_ratio="1.0"
sym="base" #"40_no_rand_2gum_no_de"
preix="wmt16_finetune"
lr_sch="fixed"
maxtoken=10000
max_epoch=40
restore_file="/home/lptang/downloads/nmt_model/scaling_nmt/wmt16.en-de.joined-dict.transformer/model.pt"

if [ ! -d "log/out/"$preix ]; then
  mkdir "log/out/"$preix
  echo "create folder log/out/"$preix
fi


name="pre_"${loss:0:2}"_"${data:0:2}"_"${model:0:2}"_"$sample"_"$sym
CUDA_VISIBLE_DEVICES=$cuda  nohup  /home/lptang/anaconda3/envs/torch/bin/fairseq-train    data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
--lr $lr --lr-scheduler $lr_sch   --weight-decay 0.0     --eval-bleu  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses     --eval-bleu-remove-bpe  --best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
--log-interval 20 --tensorboard-logdir log/tf/$preix/$name --restore-file $restore_file --reset-optimizer  \
--save-interval 1  --clip-norm 0.1 --max-epoch $max_epoch  --max-tokens $maxtoken --share-all-embeddings  --fp16 > log/out/$preix/$name.out 2>&1 &
echo "pre, "$sample", tf: "$tf_ratio", model: "${model:0:4}", "${data:0:7}", "$loss", "$name;

while :
do
    pid=$(ps -efww --sort=start_time | grep fairseq-train | grep -v grep | grep "99" | grep "00:00:0" | awk '{print $2}')
    if [ $pid ]
    then
        break
    fi
    sleep 0.5
done
echo "pid is $pid"
echo -e "$batch_size $data $loss $model $sample $lr $tf_ratio $max_epoch $lr_sch $warmup.\nTest the results of different begin checkpoint, 10, 20, 30, best from $restore_file, and tf ratio=0">>log/out/$preix/README.txt
echo -e "$pid $name\n\n">>log/out/$preix/README.txt
