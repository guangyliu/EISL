
# while :
# do
#     a=$(ps -ef |grep fairseq-train| grep -v grep)
#     b=${a:0:1}
#     if [ $b ]
#     then
#         echo "Not finish last task."
#     else
#         cd /home/lptang/fairseq/
#         bash dingshi.sh
#         echo "Start the 1st task."
#         break
#     fi
#     sleep 120
# done

# sleep 3600

while :
do
    a=$(ps -ef |grep fairseq-train| grep -v grep)
    b=${a:0:1}
    if [ $b ]
    then
        echo "Not finish the 1st task."
    else
        cd /home/lptang/fairseq/
        bash dingshi1.sh
        echo "Start the 2nd task"
        break
    fi
    sleep 120
done