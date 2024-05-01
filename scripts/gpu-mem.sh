
log_dir=../nanoGPT-LTE/log
date_str=$(date +"%Y-%m-%d")

mkdir -p $log_dir"/"$date_str

FILE=$log_dir/$date_str/gpu_mem_$(date +"%Y-%m-%d-%H-%M-%S").txt
while true; do
    echo -n "$(date +"%Y-%m-%d %H:%M:%S")" >> $FILE
    nvidia-smi | grep 'MiB /' | awk '{print substr($0, 44, 24)}' >> $FILE
    sleep 60
done