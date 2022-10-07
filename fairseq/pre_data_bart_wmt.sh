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

# DATAPATH='wmt17'
# for i in 20 25 # 30 35 40 45 50 #0 5 10 15 #
# do
#     noise="rsbs$i"
#     TASK="/home/lptang/fairseq/examples/translation/wmt17_$noise"
#     for SPLIT in train valid test
#     do
#       for LANG in en de 
#       do
#         python -m examples.roberta.multiprocessing_bpe_encoder \
#         --encoder-json /home/lptang/downloads/bart/encoder.json \
#         --vocab-bpe /home/lptang/downloads/bart/vocab.bpe \
#         --inputs "$TASK/$SPLIT.$LANG" \
#         --outputs "$TASK/$SPLIT.bpe.$LANG" \
#         --workers 20 \
#         --keep-empty &
#         sleep 3
#       done
#     done
# done
# sleep 4
# for i in 20 25 # 0 5 10 15 # 20  25 30 35 40 45 50
# do
#     noise="rsbs$i"
#     TASK="/home/lptang/fairseq/examples/translation/wmt17_$noise"
#     fairseq-preprocess \
#       --source-lang "de" \
#       --target-lang "en" \
#       --trainpref "${TASK}/train.bpe" \
#       --validpref "${TASK}/test.bpe" \
#       --testpref "${TASK}/test.bpe" \
#       --destdir "${DATAPATH}-bin/$noise" \
#       --workers 20 \
#       --srcdict /home/lptang/fairseq/examples/bart/pretrained/bart.base/dict.txt \
#       --tgtdict /home/lptang/fairseq/examples/bart/pretrained/bart.base/dict.txt & 
#       sleep 3
# done



# DATAPATH='wmt18-raw'
# # TASK="/home/lptang/fairseq/examples/translation/wmt18_raw_en_de/tmp/bart"
# for train_file in raw clean
#     do
#     for task in raw_2m raw_3m #raw_50k raw_100k raw_200k raw_500k raw_1m
#         do
#         TASK="/home/lptang/fairseq/examples/translation/wmt18_raw_en_de/"$task
#         for SPLIT in train_$train_file  test
#             do
#               for LANG in en de 
#               do
#                 python -m examples.roberta.multiprocessing_bpe_encoder \
#                 --encoder-json /home/lptang/downloads/bart/encoder.json \
#                 --vocab-bpe /home/lptang/downloads/bart/vocab.bpe \
#                 --inputs "$TASK/$SPLIT.$LANG" \
#                 --outputs "$TASK/$SPLIT.bpe.$LANG" \
#                 --workers 20 \
#                 --keep-empty 
#             #     sleep 5
#               done
#             done
#         fairseq-preprocess \
#           --source-lang "de" \
#           --target-lang "en" \
#           --trainpref "${TASK}/train_$train_file.bpe" \
#           --validpref "${TASK}/test.bpe" \
#           --testpref "${TASK}/test.bpe" \
#           --destdir "${DATAPATH}-bin/wmt18_$train_file""_$task" \
#           --workers 20 \
#           --srcdict /home/lptang/fairseq/examples/bart/pretrained/bart.base/dict.txt \
#           --tgtdict /home/lptang/fairseq/examples/bart/pretrained/bart.base/dict.txt 
#         done
#     done


DATAPATH='wmt18-raw'
# TASK="/home/lptang/fairseq/examples/translation/wmt18_raw_en_de/tmp/bart"
for train_file in raw clean
    do
    for task in raw_2m raw_3m raw_50k raw_100k raw_200k raw_500k raw_1m
        do
        TASK="/home/lptang/fairseq/examples/translation/wmt18_raw_en_de/"$task
        new_path="/home/lptang/fairseq/examples/translation/wmt18_raw_en_de/tmp"
        for SPLIT in test
            do
              for LANG in en de 
              do
                python -m examples.roberta.multiprocessing_bpe_encoder \
                --encoder-json /home/lptang/downloads/bart/encoder.json \
                --vocab-bpe /home/lptang/downloads/bart/vocab.bpe \
                --inputs "$new_path/$SPLIT.$LANG" \
                --outputs "$TASK/$SPLIT.bpe.$LANG" \
                --workers 20 \
                --keep-empty 
                echo $task
              done
            done
        fairseq-preprocess \
          --source-lang "de" \
          --target-lang "en" \
          --trainpref "${TASK}/train_$train_file.bpe" \
          --validpref "${TASK}/test.bpe" \
          --testpref "${TASK}/test.bpe" \
          --destdir "${DATAPATH}-bin/wmt18_$train_file""_$task" \
          --workers 20 \
          --srcdict /home/lptang/fairseq/examples/bart/pretrained/bart.base/dict.txt \
          --tgtdict /home/lptang/fairseq/examples/bart/pretrained/bart.base/dict.txt &
        done
    done
