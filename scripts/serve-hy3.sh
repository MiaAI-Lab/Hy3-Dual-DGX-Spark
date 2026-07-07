#!/usr/bin/env bash
# Run INSIDE the head container: docker cp this in, then `docker exec -d hy3-head bash /serve-hy3.sh`
# (see README bug #5 for why NOT to pipe it over docker exec -i)
exec > /tmp/hy3-serve.log 2>&1
export VLLM_HOST_IP=192.168.192.1
exec vllm serve /models \
  --served-model-name hy3 --port 8600 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len 131072 \
  --max-num-seqs 6 \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --trust-remote-code --enforce-eager
