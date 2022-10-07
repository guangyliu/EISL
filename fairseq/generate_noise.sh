for noise in  shuffle # rsbs # shuffle blank
do
    for loss in rl # bleu
    do
        CUDA_VISIBLE_DEVICES=2 python -u generate_noisy_MT.py --noise_list $noise --loss_list $loss &
    done

done

# for noise in rep
# do
#     for loss in rl #ce bleu
#     do
#         CUDA_VISIBLE_DEVICES=0 python -u generate_noisy_MT.py --noise_list $noise --loss_list $loss &
#     done

# done


# for noise in blank
# do
#     for loss in rl #ce bleu
#     do
#         CUDA_VISIBLE_DEVICES=3 python -u generate_noisy_MT.py --noise_list $noise --loss_list $loss &
#     done

# done