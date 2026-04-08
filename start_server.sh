#export VLLM_MOE_USE_TORCH_NAIVE=1
#export VLLM_MOE_USE_PIPELINED=1


# This is for flashinfer jit compilation
export MAX_JOBS=16
export VLLM_ENGINE_READY_TIMEOUT_S=1800
#export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_MOE_LORA_USE_PERFTE=1
export VLLM_ENABLE_MOE_DP_CHUNK=0
export CUDA_LAUNCH_BLOCKING=1
#vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 -dp 2 --gpu-memory-utilization 0.8 --enable-lora --max-lora-rank 16 --max-loras 4 --api-server-count 1 | tee main_infer_out.log
#vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 -dp 1 --gpu-memory-utilization 0.8 | tee main_ori_infer_out.log

#vllm serve Qwen/Qwen3.5-35B-A3B-FP8 --gpu-memory-utilization 0.8  --model.max-model-len 65536 --model.tool-call-parser qwen3_coder --enable-lora --max-lora-rank 16 --max-loras 4 | tee meow_infer_out.log
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 -dp 2 --gpu-memory-utilization 0.8 --enable-expert-parallel --all2all-backend flashinfer_nvlink_two_sided --api-server-count 1 --enable-lora --max-lora-rank 16 --max-loras 4 | tee main_infer_out.log
#vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 -dp 2 --gpu-memory-utilization 0.8 --enable-expert-parallel --all2all-backend allgather_reducescatter | tee main_infer_out.log

