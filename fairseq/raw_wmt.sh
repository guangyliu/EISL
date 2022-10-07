# 50k_raw 50k_clean 100k 200k 500k 1m
    cuda=3
    preix="wmt18_0904"
    num="50k"
    type="clean"
    sym=$type"_"$num
    loss="cross_entropy" # cross_entropy
    model="bart_base" # "bart_base_iter"
    tf_ratio=1.0 #'1.0'
    data="wmt18-raw-bin/wmt18_$type""_raw_$num"
    max_epoch=15
    name=${loss:0:2}"_"${model:0:2}"_"$tf_ratio"_"$sym
    TOTAL_NUM_UPDATES=30000  
    WARMUP_UPDATES=500
    LR=3e-05
    MAX_TOKENS=6000
    UPDATE_FREQ=2
    # BART_PATH=/home/lptang/fairseq/checkpoints/510_bart/cr_shuffle5_ba_1.0_s5_ce/checkpoint_best.pt

    BART_PATH=/home/lptang/fairseq/examples/bart/pretrained/bart.base/model.pt
    # BART_PATH="/home/lptang/fairseq/checkpoints/denoising_bart/$base_noise/cr_"$noise"_ba_1.0_cetf_"$i"/checkpoint_best.pt"
    BATCH_SIZE=128

    mkdir log/tf/denoising_bart_wmt/$preix
    CUDA_VISIBLE_DEVICES=$cuda  fairseq-train $data \
        --max-tokens $MAX_TOKENS \
        --task translation \
        --source-lang de --target-lang en \
        --restore-file $BART_PATH --reset-optimizer --reset-dataloader --reset-meters \
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
        --lr-scheduler inverse_sqrt --lr $LR  --warmup-updates $WARMUP_UPDATES \
        --fp16 --update-freq $UPDATE_FREQ \
        --skip-invalid-size-inputs-valid-test \
        --find-unused-parameters \
        --batch-size $BATCH_SIZE\
        --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses     --eval-bleu-remove-bpe     --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric\
        --save-dir checkpoints/denoising_bart_wmt/$preix/$name \
        --tf-ratio $tf_ratio --tensorboard-logdir log/tf/denoising_bart_wmt/$preix/$name \
        --max-epoch $max_epoch --no-epoch-checkpoints   --skip-invalid-size-inputs-valid-test  --patience 4 
#     ((cuda=cuda+1))         --reset-optimizer --reset-dataloader --reset-meters \         --restore-file $BART_PATH \ --validate-interval-updates 500  --patience 5 



    last_name=$name
    preix=$preix"_bleu"
    LR=1e-06
    loss="ngrambleuloss_nat" # cross_entropy
    model="bart_base_iter" # "bart_base_iter"
    tf_ratio=0.0
    max_epoch=10
    name=${loss:0:2}"_"${model:0:2}"_"$tf_ratio"_"$sym
    BART_PATH="/home/lptang/fairseq/checkpoints/denoising_bart_wmt/wmt18_0904/$last_name/checkpoint_best.pt"  
    TOTAL_NUM_UPDATES=15000
    mkdir log/tf/denoising_bart_wmt/$preix
    CUDA_VISIBLE_DEVICES=$cuda fairseq-train $data \
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
        --save-dir checkpoints/denoising_bart_wmt/$preix/$name \
        --tf-ratio $tf_ratio --tensorboard-logdir log/tf/denoising_bart_wmt/$preix/$name \
        --max-epoch $max_epoch --no-epoch-checkpoints --ngram 1,2 --validate-interval-updates 1000 #--validate-interval-updates 000  # --patience 5
