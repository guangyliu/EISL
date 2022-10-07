# DATAPATH='multi30k'
# for i in 0 5 10 15 20 25 30 35 40 45 50 #5 10 15 20 
# do
#     noise="mulsbs$i"
#     TASK="/home/lptang/fairseq/examples/translation/multi30k_$noise"
# #     for SPLIT in train train valid test
# #     do
# #       for LANG in en de en
# #       do
# #         python -m examples.roberta.multiprocessing_bpe_encoder \
# #         --encoder-json /home/lptang/downloads/bart/encoder.json \
# #         --vocab-bpe /home/lptang/downloads/bart/vocab.bpe \
# #         --inputs "$TASK/$SPLIT.$LANG" \
# #         --outputs "$TASK/$SPLIT.bpe.$LANG" \
# #         --workers 1 \
# #         --keep-empty ;
# #       done
# #     done


#     # mkdir "$DATAPATH-bin"
#     fairseq-preprocess \
#       --source-lang "de" \
#       --target-lang "en" \
#       --trainpref "${TASK}/train.bpe" \
#       --validpref "${TASK}/valid.bpe" \
#       --testpref "${TASK}/test.bpe" \
#       --destdir "${DATAPATH}-bin/$noise" \
#       --workers 1 \
#       --srcdict /home/lptang/fairseq/examples/bart/pretrained/bart.base/dict.txt \
#       --tgtdict /home/lptang/fairseq/examples/bart/pretrained/bart.base/dict.txt & 
# done

DATAPATH='multi30k'
for i in 0 5 10 15 20 25 30 35 40 45 50
do
    noise="rsbs$i"
    TASK="/home/lptang/fairseq/examples/translation/multi30k_$noise"
    for SPLIT in train valid test
    do
      for LANG in en de 
      do
        python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json /home/lptang/downloads/bart/encoder.json \
        --vocab-bpe /home/lptang/downloads/bart/vocab.bpe \
        --inputs "$TASK/$SPLIT.$LANG" \
        --outputs "$TASK/$SPLIT.bpe.$LANG" \
        --workers 1 \
        --keep-empty &
      done
    done
done
sleep 4
for i in 0 5 10 15 20 25 30 35 40 45 50
do
    fairseq-preprocess \
      --source-lang "de" \
      --target-lang "en" \
      --trainpref "${TASK}/train.bpe" \
      --validpref "${TASK}/valid.bpe" \
      --testpref "${TASK}/test.bpe" \
      --destdir "${DATAPATH}-bin/$noise" \
      --workers 1 \
      --srcdict /home/lptang/fairseq/examples/bart/pretrained/bart.base/dict.txt \
      --tgtdict /home/lptang/fairseq/examples/bart/pretrained/bart.base/dict.txt & 
done