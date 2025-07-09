# start_servers.sh
#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <worker_id>"
  exit 1
fi

WORKER_ID=$1

LOG_DIR="app_logs/${WORKER_ID}"
mkdir -p "${LOG_DIR}"

for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i nohup python embedding_app.py --ip 0.0.0.0 --port 800$i > "${LOG_DIR}/log_$i.log" 2>&1 &
    echo "Started server on GPU $i at port 800$i"
done

