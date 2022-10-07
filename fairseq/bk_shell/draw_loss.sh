#!/bin/bash

cuda=0
batch_size="512"
# data: iwslt14.tokenized.de-en    multi30k.de-en

# loss: cross_entropy  bleuloss ngrambleuloss_nat
loss="ngrambleuloss_drawloss"
# model: transformer_iwslt_de_en   lstm_wiseman_iwslt_de_en_gumbel
model="lstm_wiseman_iwslt_de_en_gumbel"
sample="greedy"
lr="1e-3"
max_epoch=50
noise="mulsirb"
preix="510_loss_"$noise
lr_sch="fixed" #
# warmup=2000
maxtoken=6000
restore_file="checkpoints/lstm_multi30k/cr_mu_ls_greedy_base/checkpoint_best.pt"  # "checkpoints/base/bl_iw_base_5/checkpoint_best.pt"
if [ ! -d "log/out/"$preix ]; then
  mkdir "log/out/"$preix
  echo "crete folder log/out/"$preix
fi
for tf_ratio in "1.0" "0.0"
do
    for sym in 0 5 10 15 20 25 30 35 40 45 50   #0 5 10 15 20 25 30 35 40 45 50  # 0 4 8 12 16 # 0 5 10 15 20 25 30 35 40 45 50 #2 3 4 5 6 7 8 9 10 all
    do
        data="multi30k_"$noise$sym
        echo $data
        name=$tf_ratio"_"$sym
        CUDA_VISIBLE_DEVICES=$cuda  fairseq-train    data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
            --lr $lr --lr-scheduler $lr_sch  --weight-decay 0.0001     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
            --eval-bleu-detok moses     --eval-bleu-remove-bpe   --eval-bleu-print-samples  --best-checkpoint-metric bleu \
            --maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
            --log-interval 1 --tensorboard-logdir log/tf/$preix/$name --sample-method $sample --tf-ratio $tf_ratio \
            --save-interval 1 --max-epoch $max_epoch --clip-norm 0.1       --max-tokens $maxtoken    --restore-file $restore_file  --reset-optimizer  &
        sleep 2.5
    done
done

# #!/bin/sh
# cuda=3
# batch_size="256"
# # data: iwslt14.tokenized.de-en    multi30k_de_en
# data="iwslt14.tokenized.de-en"
# # loss: cross_entropy  bleuloss cebleuloss ngrambleuloss ngrambleulossNAT
# loss="ngrambleuloss_nat"
# # model: transformer_iwslt_de_en   lstm_wiseman_iwslt_de_en_gumbel  
# model="lstm_wiseman_iwslt_de_en_gumbel"
# # restore-file if use pretrained model
# pre=1

# # sample 
# sample="greedy"
# top_k=40
# # lr
# lr="1e-3"
# tf_ratio="0.8"
# sym="debug" #"40_no_rand_2gum_no_de"
# preix="debug"
# lr_sch="fixed"
# warmup=3000
# max_epoch=40
# restore_file="checkpoints/base/cr_iw_ls_greedy_realce_tf10_test/checkpoint_best.pt"

# if [ ! -d "log/out/"$preix ]; then
#   mkdir "log/out/"$preix
#   echo "create folder log/out/"$preix
# fi

# if [ $pre == 1 ]
# then 
#     name="pre_"${loss:0:2}"_"${data:0:2}"_"${model:0:2}"_"$sample"_"$sym
#     CUDA_VISIBLE_DEVICES=$cuda  nohup  /home/lptang/anaconda3/envs/torch/bin/fairseq-train    data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
#     --lr $lr --lr-scheduler $lr_sch   --weight-decay 0.0001     --eval-bleu  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses     --eval-bleu-remove-bpe   --eval-bleu-print-samples  --best-checkpoint-metric bleu \
#     --maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
#     --log-interval 100 --tensorboard-logdir log/tf/$preix/$name --restore-file $restore_file --reset-optimizer --sample-method $sample --tf-ratio $tf_ratio \
#     --save-interval 1  --clip-norm 0.1 --max-epoch $max_epoch   --top-k $top_k  > log/out/$preix/$name.out 2>&1 &
#     echo "pre, "$sample", tf: "$tf_ratio", model: "${model:0:4}", "${data:0:7}", "$loss", "$name;
# else
#     name=${loss:0:2}"_"${data:0:2}"_"${model:0:2}"_"$sample"_"$sym
#     CUDA_VISIBLE_DEVICES=$cuda nohup /home/lptang/anaconda3/envs/torch/bin/fairseq-train     data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
#     --lr $lr --lr-scheduler $lr_sch  --weight-decay 0.0001     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses     --eval-bleu-remove-bpe   --eval-bleu-print-samples  --best-checkpoint-metric bleu \
#     --maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
#     --log-interval 20 --tensorboard-logdir log/tf/$preix/$name --sample-method $sample --tf-ratio $tf_ratio \
#     --save-interval 1 --clip-norm 0.1 --max-epoch $max_epoch  --top-k $top_k > log/out/$preix/$name.out 2>&1 &
#     echo "not pre, " $sample", tf: "$tf_ratio", model: "${model:0:4}", "${data:0:7}", "$loss", "$name;
# fi

# while :
# do
#     pid=$(ps -efww --sort=start_time | grep fairseq-train | grep -v grep | grep "99" | grep "00:00:0" | awk '{print $2}')
#     if [ $pid ]
#     then
#         break
#     fi
#     sleep 0.5
# done
# echo "pid is $pid"
# echo -e "$batch_size $data $loss $model $sample $lr $tf_ratio $max_epoch $lr_sch $warmup.\nTest the results of different begin checkpoint, 10, 20, 30, best from $restore_file, and tf ratio=0">>log/out/$preix/README.txt
# echo -e "$pid $name\n\n">>log/out/$preix/README.txt
