#!/bin/sh
cuda=0
tf_ratio="1.0"
sym="from_n1_tf08_to_n_2_tf10"
batch_size="256"
# data: iwslt14.tokenized.de-en    multi30k_de_en
data="iwslt14.tokenized.de-en"
# loss: cross_entropy  bleuloss cebleuloss ngrambleuloss
loss="ngrambleuloss"
# model: transformer_iwslt_de_en   lstm_wiseman_iwslt_de_en_gumbel
model="lstm_wiseman_iwslt_de_en_gumbel"
# restore-file if use pretrained model
pre=1

# sample 
sample="greedy"
# lr
lr="1e-3"
max_epoch=45
preix="n_1_to_n_2"
lr_sch="fixed"
# warmup=1000
restore_file="checkpoints/pre_n/pre_ng_iw_ls_greedy_from_bestn_to_n_1_tf08_v2/checkpoint_best.pt"

if [ ! -d "log/out/"$preix ]; then
  mkdir "log/out/"$preix
  echo "create folder log/out/"$preix
fi


# name="pre_"${loss:0:2}"_"${data:0:2}"_"${model:0:2}"_"$sample"_"$sym
# CUDA_VISIBLE_DEVICES=$cuda  nohup fairseq-train    data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
# --lr $lr --lr-scheduler $lr_sch   --weight-decay 0.0001     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
# --eval-bleu-detok moses     --eval-bleu-remove-bpe   --eval-bleu-print-samples  --best-checkpoint-metric bleu \
# --maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
# --log-interval 20 --tensorboard-logdir log/tf/$preix/$name --restore-file $restore_file --reset-optimizer --sample-method $sample --tf-ratio $tf_ratio \
# --save-interval 1  --clip-norm 0.1 --max-epoch $max_epoch > log/out/$preix/$name.out 2>&1 &
# echo "pre, "$sample", tf: "$tf_ratio", model: "${model:0:4}", "${data:0:7}", "$loss", "$name;


# while :
# do
#     pid=$(ps -ef --sort=start_time | grep fairseq-train | grep -v grep | grep "99" | grep "00:00:" | awk '{print $2}')
#     if [ $pid ]
#     then
#         break
#     fi
#     sleep 0.5
# done
# echo "pid is $pid"
# echo -e "$batch_size $data $loss $model $sample $lr $tf_ratio $max_epoch $lr_sch $warmup.\nTest the results of different begin checkpoint, 10, 20, 30, best from $restore_file, and tf ratio=0">>log/out/$preix/README.txt
# echo -e "$pid $name\n\n">>log/out/$preix/README.txt

# sleep 10
# cuda=0
# tf_ratio="0.9"
# sym="from_n1_tf08_to_n_2_tf09"
# name="pre_"${loss:0:2}"_"${data:0:2}"_"${model:0:2}"_"$sample"_"$sym
# CUDA_VISIBLE_DEVICES=$cuda  nohup fairseq-train    data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
# --lr $lr --lr-scheduler $lr_sch   --weight-decay 0.0001     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
# --eval-bleu-detok moses     --eval-bleu-remove-bpe   --eval-bleu-print-samples  --best-checkpoint-metric bleu \
# --maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
# --log-interval 20 --tensorboard-logdir log/tf/$preix/$name --restore-file $restore_file --reset-optimizer --sample-method $sample --tf-ratio $tf_ratio \
# --save-interval 1  --clip-norm 0.1 --max-epoch $max_epoch > log/out/$preix/$name.out 2>&1 &
# echo "pre, "$sample", tf: "$tf_ratio", model: "${model:0:4}", "${data:0:7}", "$loss", "$name;
# while :
# do
#     pid=$(ps -ef --sort=start_time | grep fairseq-train | grep -v grep | grep "99" | grep "00:00:" | awk '{print $2}')
#     if [ $pid ]
#     then
#         break
#     fi
#     sleep 0.5
# done
# echo "pid is $pid"
# echo -e "$batch_size $data $loss $model $sample $lr $tf_ratio $max_epoch $lr_sch $warmup.\nTest the results of different begin checkpoint, 10, 20, 30, best from $restore_file, and tf ratio=0">>log/out/$preix/README.txt
# echo -e "$pid $name\n\n">>log/out/$preix/README.txt


# sleep 10
# cuda=1
# tf_ratio="0.8"
# sym="from_n1_tf08_to_n_2_tf08"
# name="pre_"${loss:0:2}"_"${data:0:2}"_"${model:0:2}"_"$sample"_"$sym
# CUDA_VISIBLE_DEVICES=$cuda  nohup fairseq-train    data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
# --lr $lr --lr-scheduler $lr_sch   --weight-decay 0.0001     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
# --eval-bleu-detok moses     --eval-bleu-remove-bpe   --eval-bleu-print-samples  --best-checkpoint-metric bleu \
# --maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
# --log-interval 20 --tensorboard-logdir log/tf/$preix/$name --restore-file $restore_file --reset-optimizer --sample-method $sample --tf-ratio $tf_ratio \
# --save-interval 1  --clip-norm 0.1 --max-epoch $max_epoch > log/out/$preix/$name.out 2>&1 &
# echo "pre, "$sample", tf: "$tf_ratio", model: "${model:0:4}", "${data:0:7}", "$loss", "$name;
# while :
# do
#     pid=$(ps -ef --sort=start_time | grep fairseq-train | grep -v grep | grep "99" | grep "00:00:" | awk '{print $2}')
#     if [ $pid ]
#     then
#         break
#     fi
#     sleep 0.5
# done
# echo "pid is $pid"
# echo -e "$batch_size $data $loss $model $sample $lr $tf_ratio $max_epoch $lr_sch $warmup.\nTest the results of different begin checkpoint, 10, 20, 30, best from $restore_file, and tf ratio=0">>log/out/$preix/README.txt
# echo -e "$pid $name\n\n">>log/out/$preix/README.txt


# sleep 10
# cuda=1
# tf_ratio="0.7"
# sym="from_n1_tf08_to_n_2_tf07"
# name="pre_"${loss:0:2}"_"${data:0:2}"_"${model:0:2}"_"$sample"_"$sym
# CUDA_VISIBLE_DEVICES=$cuda  nohup fairseq-train    data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
# --lr $lr --lr-scheduler $lr_sch   --weight-decay 0.0001     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
# --eval-bleu-detok moses     --eval-bleu-remove-bpe   --eval-bleu-print-samples  --best-checkpoint-metric bleu \
# --maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
# --log-interval 20 --tensorboard-logdir log/tf/$preix/$name --restore-file $restore_file --reset-optimizer --sample-method $sample --tf-ratio $tf_ratio \
# --save-interval 1  --clip-norm 0.1 --max-epoch $max_epoch > log/out/$preix/$name.out 2>&1 &
# echo "pre, "$sample", tf: "$tf_ratio", model: "${model:0:4}", "${data:0:7}", "$loss", "$name;
# while :
# do
#     pid=$(ps -ef --sort=start_time | grep fairseq-train | grep -v grep | grep "99" | grep "00:00:" | awk '{print $2}')
#     if [ $pid ]
#     then
#         break
#     fi
#     sleep 0.5
# done
# echo "pid is $pid"
# echo -e "$batch_size $data $loss $model $sample $lr $tf_ratio $max_epoch $lr_sch $warmup.\nTest the results of different begin checkpoint, 10, 20, 30, best from $restore_file, and tf ratio=0">>log/out/$preix/README.txt
# echo -e "$pid $name\n\n">>log/out/$preix/README.txt


# sleep 10
# cuda=2
# tf_ratio="0.6"
# sym="from_n1_tf08_to_n_2_tf06"
# name="pre_"${loss:0:2}"_"${data:0:2}"_"${model:0:2}"_"$sample"_"$sym
# CUDA_VISIBLE_DEVICES=$cuda  nohup fairseq-train    data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
# --lr $lr --lr-scheduler $lr_sch   --weight-decay 0.0001     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
# --eval-bleu-detok moses     --eval-bleu-remove-bpe   --eval-bleu-print-samples  --best-checkpoint-metric bleu \
# --maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
# --log-interval 20 --tensorboard-logdir log/tf/$preix/$name --restore-file $restore_file --reset-optimizer --sample-method $sample --tf-ratio $tf_ratio \
# --save-interval 1  --clip-norm 0.1 --max-epoch $max_epoch > log/out/$preix/$name.out 2>&1 &
# echo "pre, "$sample", tf: "$tf_ratio", model: "${model:0:4}", "${data:0:7}", "$loss", "$name;
# while :
# do
#     pid=$(ps -ef --sort=start_time | grep fairseq-train | grep -v grep | grep "99" | grep "00:00:" | awk '{print $2}')
#     if [ $pid ]
#     then
#         break
#     fi
#     sleep 0.5
# done
# echo "pid is $pid"
# echo -e "$batch_size $data $loss $model $sample $lr $tf_ratio $max_epoch $lr_sch $warmup.\nTest the results of different begin checkpoint, 10, 20, 30, best from $restore_file, and tf ratio=0">>log/out/$preix/README.txt
# echo -e "$pid $name\n\n">>log/out/$preix/README.txt


# sleep 10
# cuda=2
# tf_ratio="0.5"
# sym="from_n1_tf08_to_n_2_tf05"
# name="pre_"${loss:0:2}"_"${data:0:2}"_"${model:0:2}"_"$sample"_"$sym
# CUDA_VISIBLE_DEVICES=$cuda  nohup fairseq-train    data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
# --lr $lr --lr-scheduler $lr_sch   --weight-decay 0.0001     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
# --eval-bleu-detok moses     --eval-bleu-remove-bpe   --eval-bleu-print-samples  --best-checkpoint-metric bleu \
# --maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
# --log-interval 20 --tensorboard-logdir log/tf/$preix/$name --restore-file $restore_file --reset-optimizer --sample-method $sample --tf-ratio $tf_ratio \
# --save-interval 1  --clip-norm 0.1 --max-epoch $max_epoch > log/out/$preix/$name.out 2>&1 &
# echo "pre, "$sample", tf: "$tf_ratio", model: "${model:0:4}", "${data:0:7}", "$loss", "$name;
# while :
# do
#     pid=$(ps -ef --sort=start_time | grep fairseq-train | grep -v grep | grep "99" | grep "00:00:" | awk '{print $2}')
#     if [ $pid ]
#     then
#         break
#     fi
#     sleep 0.5
# done
# echo "pid is $pid"
# echo -e "$batch_size $data $loss $model $sample $lr $tf_ratio $max_epoch $lr_sch $warmup.\nTest the results of different begin checkpoint, 10, 20, 30, best from $restore_file, and tf ratio=0">>log/out/$preix/README.txt
# echo -e "$pid $name\n\n">>log/out/$preix/README.txt


# sleep 10
cuda=3
tf_ratio="0.4"
sym="from_n1_tf08_to_n_2_tf04"
name="pre_"${loss:0:2}"_"${data:0:2}"_"${model:0:2}"_"$sample"_"$sym
CUDA_VISIBLE_DEVICES=$cuda  nohup fairseq-train    data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
--lr $lr --lr-scheduler $lr_sch   --weight-decay 0.0001     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses     --eval-bleu-remove-bpe   --eval-bleu-print-samples  --best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
--log-interval 20 --tensorboard-logdir log/tf/$preix/$name --restore-file $restore_file --reset-optimizer --sample-method $sample --tf-ratio $tf_ratio \
--save-interval 1  --clip-norm 0.1 --max-epoch $max_epoch > log/out/$preix/$name.out 2>&1 &
echo "pre, "$sample", tf: "$tf_ratio", model: "${model:0:4}", "${data:0:7}", "$loss", "$name;
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


sleep 10
cuda=3
tf_ratio="0.3"
sym="from_n1_tf08_to_n_2_tf03"
name="pre_"${loss:0:2}"_"${data:0:2}"_"${model:0:2}"_"$sample"_"$sym
CUDA_VISIBLE_DEVICES=$cuda  nohup fairseq-train    data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
--lr $lr --lr-scheduler $lr_sch   --weight-decay 0.0001     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses     --eval-bleu-remove-bpe   --eval-bleu-print-samples  --best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
--log-interval 20 --tensorboard-logdir log/tf/$preix/$name --restore-file $restore_file --reset-optimizer --sample-method $sample --tf-ratio $tf_ratio \
--save-interval 1  --clip-norm 0.1 --max-epoch $max_epoch > log/out/$preix/$name.out 2>&1 &
echo "pre, "$sample", tf: "$tf_ratio", model: "${model:0:4}", "${data:0:7}", "$loss", "$name;
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
