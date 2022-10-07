# noise=shuffle
# for i in 2 3 4 5 6 7 8 9 10
# do
#     TEXT=examples/translation/multi30k_$noise$i
#     fairseq-preprocess \
#         --source-lang de --target-lang en \
#         --trainpref $TEXT/test --validpref $TEXT/valid --testpref $TEXT/test \
#         --destdir data-bin/valid/multi30k_$noise$i --thresholdtgt 0 --thresholdsrc 0 \
#         --workers 20 --tgtdict data-bin/multi30k.de-en/dict.en.txt --srcdict data-bin/multi30k.de-en/dict.de.txt &
#         sleep 0.3
# done
# noise=blank
# for i in 1 2 3 4 5 6 7 8 9
# do
#     TEXT=examples/translation/multi30k_$noise$i
#     fairseq-preprocess \
#         --source-lang de --target-lang en \
#         --trainpref $TEXT/test --validpref $TEXT/valid --testpref $TEXT/test \
#         --destdir data-bin/valid/multi30k_$noise$i --thresholdtgt 0 --thresholdsrc 0 \
#         --workers 20 --tgtdict data-bin/multi30k.de-en/dict.en.txt --srcdict data-bin/multi30k.de-en/dict.de.txt &
#         sleep 0.3
# done
noise=mulsirb
for i in 0 5 10 15 20 25 30 35 40 45 50 #0 5 10 15 20 25 30 35 40 45 50  #0 1 2 3 4 5 6 7 8 9  #
do
    TEXT=examples/translation/bpe_multi30k_$noise$i
    fairseq-preprocess \
        --source-lang de --target-lang en \
        --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
        --destdir data-bin/multi30k_$noise$i --thresholdtgt 0 --thresholdsrc 0 \
        --workers 1 --tgtdict data-bin/multi30k.de-en/dict.en.txt --srcdict data-bin/multi30k.de-en/dict.de.txt&
        sleep 0.3
    
done