for noise in rsbs # blank shuffle rep blank # rsbs
do
    cd $noise
    for file in `ls eval_rl*`
    do
        echo "$noise __ $file:"
        cat $file|awk '{sum+=$1} END {print "Average = ", sum/NR}'
    done
    cd ..
done