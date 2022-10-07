for i in 5 10 15 20 25 30 35 40 45 50 #20 #10 15 20 25 30 35 40 45 50 # 35 50 # 5 15 30 #  10 20 25 # # 
do
base_noise="rsbs"
preix=$base_noise
sym="cetf_$i"
cuda=0
loss="cross_entropy" # cross_entropy
model="bart_base" # "bart_base_iter"
tf_ratio=1.0 #'1.0'
noise="$preix$i"
data="multi30k-bin/$noise"
max_epoch=20
name=${loss:0:2}"_"${noise}"_"${model:0:2}"_"$tf_ratio"_"$sym
TOTAL_NUM_UPDATES=10000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=6000
UPDATE_FREQ=4
# BART_PATH=/home/lptang/fairseq/checkpoints/510_bart/cr_shuffle5_ba_1.0_s5_ce/checkpoint_best.pt

BART_PATH=/home/lptang/fairseq/examples/bart/pretrained/bart.base/model.pt
# BART_PATH="/home/lptang/fairseq/checkpoints/denoising_bart/$base_noise/cr_"$noise"_ba_1.0_cetf_"$i"/checkpoint_best.pt"
BATCH_SIZE=128
if [ ! -d "log/out/"$preix ]; then
  mkdir "log/out/"$preix
  echo "crete folder log/out/"$preix
fi
mkdir log/tf/denoising_bart/$preix
# CUDA_VISIBLE_DEVICES=$cuda  fairseq-train multi30k-bin/$noise \
#     --restore-file $BART_PATH \
#     --max-tokens $MAX_TOKENS \
#     --task translation \
#     --source-lang de --target-lang en \
#     --truncate-source \
#     --layernorm-embedding \
#     --share-all-embeddings \
#     --share-decoder-input-output-embed \
#     --reset-optimizer --reset-dataloader --reset-meters \
#     --required-batch-size-multiple 1 \
#     --arch $model \
#     --criterion $loss \
#     --dropout 0.1 --attention-dropout 0.1 \
#     --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
#     --clip-norm 0.1 \
#     --lr-scheduler polynomial_decay --lr $LR  --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES\
#     --fp16 --update-freq $UPDATE_FREQ \
#     --skip-invalid-size-inputs-valid-test \
#     --find-unused-parameters\
#     --batch-size $BATCH_SIZE\
#     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses     --eval-bleu-remove-bpe     --best-checkpoint-metric bleu \
#     --maximize-best-checkpoint-metric\
#     --save-dir checkpoints/denoising_bart/$preix/$name \
#     --tf-ratio $tf_ratio --tensorboard-logdir log/tf/denoising_bart/$preix/$name \
#      --max-epoch $max_epoch --no-epoch-checkpoints   --patience 3 #--eval-bleu-print-samples  --log-interval 5
     
#     preix=$base_noise"_bleu"
#     sym="1118_bleu12_$i"
#     loss="ngrambleuloss_nat" # cross_entropy
#     model="bart_base_iter" # "bart_base_iter"
#     tf_ratio=0.0
#     noise="$base_noise$i"
#     data="multi30k-bin/$noise"
#     max_epoch=15
#     name=${loss:0:2}"_"${noise}"_"${model:0:2}"_"$tf_ratio"_"$sym
#     TOTAL_NUM_UPDATES=10000  
#     WARMUP_UPDATES=500      
#     LR=1e-05
#     MAX_TOKENS=6000
#     UPDATE_FREQ=4
# #     BART_PATH=/home/lptang/fairseq/examples/bart/pretrained/bart.base/model.pt
#     BART_PATH="/home/lptang/fairseq/checkpoints/denoising_bart/$base_noise/cr_"$noise"_ba_1.0_cetf_"$i"/checkpoint_best.pt"
#     #"/home/lptang/fairseq/checkpoints/denoising_bart/shuffle/cr_shuffle"$i"_ba_1.0_cetf_"$i"/checkpoint_best.pt"
#     BATCH_SIZE=128
#     if [ ! -d "log/out/"$preix ]; then
#       mkdir "log/out/"$preix
#       echo "crete folder log/out/"$preix
#     fi
#     CUDA_VISIBLE_DEVICES=$cuda fairseq-train multi30k-bin/$noise \
#         --restore-file $BART_PATH --reset-optimizer --reset-dataloader --reset-meters \
#         --max-tokens $MAX_TOKENS \
#         --task translation \
#         --source-lang de --target-lang en \
#         --truncate-source \
#         --layernorm-embedding \
#         --share-all-embeddings \
#         --share-decoder-input-output-embed \
#         --required-batch-size-multiple 1 \
#         --arch $model \
#         --criterion $loss \
#         --dropout 0.1 --attention-dropout 0.1 \
#         --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
#         --clip-norm 0.1 \
#         --lr-scheduler polynomial_decay --lr $LR  --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES\
#         --fp16 --update-freq $UPDATE_FREQ \
#         --skip-invalid-size-inputs-valid-test \
#         --find-unused-parameters\
#         --batch-size $BATCH_SIZE\
#         --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#         --eval-bleu-detok moses     --eval-bleu-remove-bpe    --best-checkpoint-metric bleu \
#         --maximize-best-checkpoint-metric\
#         --save-dir checkpoints/denoising_bart/$preix/$name \
#         --tf-ratio $tf_ratio --tensorboard-logdir log/tf/denoising_bart/$preix/$name \
#         --max-epoch $max_epoch --no-epoch-checkpoints --patience 4
        
        
    preix=$base_noise"_rl"
    sym="1118rl_$i"
    loss="reward_cross_entropy" # cross_entropy
    model="bart_base" # "bart_base_iter"
    tf_ratio=1.0
    noise="$base_noise$i"
    data="multi30k-bin/$noise"
    max_epoch=20
    name=${loss:0:2}"_"${noise}"_"${model:0:2}"_"$tf_ratio"_"$sym
    TOTAL_NUM_UPDATES=10000  
    WARMUP_UPDATES=500      
    LR=3e-05
    MAX_TOKENS=6000
    UPDATE_FREQ=4
#     BART_PATH=/home/lptang/fairseq/examples/bart/pretrained/bart.base/model.pt
    BART_PATH="/home/lptang/fairseq/checkpoints/denoising_bart/$base_noise/cr_"$noise"_ba_1.0_cetf_"$i"/checkpoint_best.pt"
    #"/home/lptang/fairseq/checkpoints/denoising_bart/shuffle/cr_shuffle"$i"_ba_1.0_cetf_"$i"/checkpoint_best.pt"
    BATCH_SIZE=128
    if [ ! -d "log/out/"$preix ]; then
      mkdir "log/out/"$preix
      echo "crete folder log/out/"$preix
    fi
    CUDA_VISIBLE_DEVICES=$cuda fairseq-train multi30k-bin/$noise \
        --restore-file $BART_PATH --reset-optimizer --reset-dataloader --reset-meters \
        --max-tokens $MAX_TOKENS \
        --task translation \
        --source-lang de --target-lang en \
        --truncate-source \
        --layernorm-embedding \
        --share-all-embeddings \
        --share-decoder-input-output-embed \
        --required-batch-size-multiple 1 \
        --arch $model \
        --criterion $loss \
        --dropout 0.1 --attention-dropout 0.1 \
        --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
        --clip-norm 0.1 \
        --lr-scheduler polynomial_decay --lr $LR  --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES\
        --fp16 --update-freq $UPDATE_FREQ \
        --skip-invalid-size-inputs-valid-test \
        --find-unused-parameters\
        --batch-size $BATCH_SIZE\
        --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses     --eval-bleu-remove-bpe    --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric\
        --save-dir checkpoints/denoising_bart/$preix/$name \
        --tf-ratio $tf_ratio --tensorboard-logdir log/tf/denoising_bart/$preix/$name \
        --max-epoch $max_epoch --no-epoch-checkpoints --patience 2
done



for i in 2 3 4 5 6 7 8 9 10 all 0 #20 #10 15 20 25 30 35 40 45 50 # 35 50 # 5 15 30 #  10 20 25 # # 
do
base_noise="shuffle"
preix=$base_noise
sym="cetf_$i"
cuda=0
loss="cross_entropy" # cross_entropy
model="bart_base" # "bart_base_iter"
tf_ratio=1.0 #'1.0'
noise="$preix$i"
data="multi30k-bin/$noise"
max_epoch=20
name=${loss:0:2}"_"${noise}"_"${model:0:2}"_"$tf_ratio"_"$sym
TOTAL_NUM_UPDATES=10000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=6000
UPDATE_FREQ=4
# BART_PATH=/home/lptang/fairseq/checkpoints/510_bart/cr_shuffle5_ba_1.0_s5_ce/checkpoint_best.pt

BART_PATH=/home/lptang/fairseq/examples/bart/pretrained/bart.base/model.pt
# BART_PATH="/home/lptang/fairseq/checkpoints/denoising_bart/$base_noise/cr_"$noise"_ba_1.0_cetf_"$i"/checkpoint_best.pt"
BATCH_SIZE=128
if [ ! -d "log/out/"$preix ]; then
  mkdir "log/out/"$preix
  echo "crete folder log/out/"$preix
fi
mkdir log/tf/denoising_bart/$preix
# CUDA_VISIBLE_DEVICES=$cuda  fairseq-train multi30k-bin/$noise \
#     --restore-file $BART_PATH \
#     --max-tokens $MAX_TOKENS \
#     --task translation \
#     --source-lang de --target-lang en \
#     --truncate-source \
#     --layernorm-embedding \
#     --share-all-embeddings \
#     --share-decoder-input-output-embed \
#     --reset-optimizer --reset-dataloader --reset-meters \
#     --required-batch-size-multiple 1 \
#     --arch $model \
#     --criterion $loss \
#     --dropout 0.1 --attention-dropout 0.1 \
#     --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
#     --clip-norm 0.1 \
#     --lr-scheduler polynomial_decay --lr $LR  --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES\
#     --fp16 --update-freq $UPDATE_FREQ \
#     --skip-invalid-size-inputs-valid-test \
#     --find-unused-parameters\
#     --batch-size $BATCH_SIZE\
#     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses     --eval-bleu-remove-bpe     --best-checkpoint-metric bleu \
#     --maximize-best-checkpoint-metric\
#     --save-dir checkpoints/denoising_bart/$preix/$name \
#     --tf-ratio $tf_ratio --tensorboard-logdir log/tf/denoising_bart/$preix/$name \
#      --max-epoch $max_epoch --no-epoch-checkpoints   --patience 3 #--eval-bleu-print-samples  --log-interval 5
     
#     preix=$base_noise"_bleu"
#     sym="1118_bleu12_$i"
#     loss="ngrambleuloss_nat" # cross_entropy
#     model="bart_base_iter" # "bart_base_iter"
#     tf_ratio=0.0
#     noise="$base_noise$i"
#     data="multi30k-bin/$noise"
#     max_epoch=15
#     name=${loss:0:2}"_"${noise}"_"${model:0:2}"_"$tf_ratio"_"$sym
#     TOTAL_NUM_UPDATES=10000  
#     WARMUP_UPDATES=500      
#     LR=1e-05
#     MAX_TOKENS=6000
#     UPDATE_FREQ=4
# #     BART_PATH=/home/lptang/fairseq/examples/bart/pretrained/bart.base/model.pt
#     BART_PATH="/home/lptang/fairseq/checkpoints/denoising_bart/$base_noise/cr_"$noise"_ba_1.0_cetf_"$i"/checkpoint_best.pt"
#     #"/home/lptang/fairseq/checkpoints/denoising_bart/shuffle/cr_shuffle"$i"_ba_1.0_cetf_"$i"/checkpoint_best.pt"
#     BATCH_SIZE=128
#     if [ ! -d "log/out/"$preix ]; then
#       mkdir "log/out/"$preix
#       echo "crete folder log/out/"$preix
#     fi
#     CUDA_VISIBLE_DEVICES=$cuda fairseq-train multi30k-bin/$noise \
#         --restore-file $BART_PATH --reset-optimizer --reset-dataloader --reset-meters \
#         --max-tokens $MAX_TOKENS \
#         --task translation \
#         --source-lang de --target-lang en \
#         --truncate-source \
#         --layernorm-embedding \
#         --share-all-embeddings \
#         --share-decoder-input-output-embed \
#         --required-batch-size-multiple 1 \
#         --arch $model \
#         --criterion $loss \
#         --dropout 0.1 --attention-dropout 0.1 \
#         --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
#         --clip-norm 0.1 \
#         --lr-scheduler polynomial_decay --lr $LR  --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES\
#         --fp16 --update-freq $UPDATE_FREQ \
#         --skip-invalid-size-inputs-valid-test \
#         --find-unused-parameters\
#         --batch-size $BATCH_SIZE\
#         --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#         --eval-bleu-detok moses     --eval-bleu-remove-bpe    --best-checkpoint-metric bleu \
#         --maximize-best-checkpoint-metric\
#         --save-dir checkpoints/denoising_bart/$preix/$name \
#         --tf-ratio $tf_ratio --tensorboard-logdir log/tf/denoising_bart/$preix/$name \
#         --max-epoch $max_epoch --no-epoch-checkpoints --patience 4
        
        
    preix=$base_noise"_rl"
    sym="1118rl_$i"
    loss="reward_cross_entropy" # cross_entropy
    model="bart_base" # "bart_base_iter"
    tf_ratio=1.0
    noise="$base_noise$i"
    data="multi30k-bin/$noise"
    max_epoch=20
    name=${loss:0:2}"_"${noise}"_"${model:0:2}"_"$tf_ratio"_"$sym
    TOTAL_NUM_UPDATES=10000  
    WARMUP_UPDATES=500      
    LR=3e-05
    MAX_TOKENS=6000
    UPDATE_FREQ=4
#     BART_PATH=/home/lptang/fairseq/examples/bart/pretrained/bart.base/model.pt
    BART_PATH="/home/lptang/fairseq/checkpoints/denoising_bart/$base_noise/cr_"$noise"_ba_1.0_cetf_"$i"/checkpoint_best.pt"
    #"/home/lptang/fairseq/checkpoints/denoising_bart/shuffle/cr_shuffle"$i"_ba_1.0_cetf_"$i"/checkpoint_best.pt"
    BATCH_SIZE=128
    if [ ! -d "log/out/"$preix ]; then
      mkdir "log/out/"$preix
      echo "crete folder log/out/"$preix
    fi
    CUDA_VISIBLE_DEVICES=$cuda fairseq-train multi30k-bin/$noise \
        --restore-file $BART_PATH --reset-optimizer --reset-dataloader --reset-meters \
        --max-tokens $MAX_TOKENS \
        --task translation \
        --source-lang de --target-lang en \
        --truncate-source \
        --layernorm-embedding \
        --share-all-embeddings \
        --share-decoder-input-output-embed \
        --required-batch-size-multiple 1 \
        --arch $model \
        --criterion $loss \
        --dropout 0.1 --attention-dropout 0.1 \
        --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
        --clip-norm 0.1 \
        --lr-scheduler polynomial_decay --lr $LR  --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES\
        --fp16 --update-freq $UPDATE_FREQ \
        --skip-invalid-size-inputs-valid-test \
        --find-unused-parameters\
        --batch-size $BATCH_SIZE\
        --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses     --eval-bleu-remove-bpe    --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric\
        --save-dir checkpoints/denoising_bart/$preix/$name \
        --tf-ratio $tf_ratio --tensorboard-logdir log/tf/denoising_bart/$preix/$name \
        --max-epoch $max_epoch --no-epoch-checkpoints --patience 2
done

