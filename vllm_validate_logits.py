import time
import argparse
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.executor.mp_distributed_executor import MultiprocessingDistributedExecutor
from meow_sampler import generate_neg_gumbel_noise
#from toploc import verify_proofs_bytes

# Force multiproc to spawn
#import torch.multiprocessing
#torch.multiprocessing.set_start_method("spawn")

# Global constants
TOPK: int = 5
MAX_CHUNK_SIZE: int = 512  # Maximum number of tokens to process in each chunk

processed_count = 0
results = None

def parse_args():
    parser = argparse.ArgumentParser(description="Run validation on model activations and commits.")

    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save outputs and commits.")
    parser.add_argument("--decode_model_name", type=str, required=True, help="Model name used for decoding.")
    parser.add_argument("--validate_model_name", type=str, required=True, help="Model name used for validation.")
    parser.add_argument("--device_str", type=str, required=True, help="Used by the savefile")
    parser.add_argument("--max_decode_tokens", type=int, default=8000, help="Maximum number of decode tokens.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for activations.")
    parser.add_argument("--attn", type=str, default="flash", help="Attention implementation for the model.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--decode_kv_cache_dtype", type=str, default="auto", 
                        choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
                        help="KV cache dtype used in the decode model. Used only for filename construction.")
    parser.add_argument("--toploc_topk", type=int, default=128,
                        help="Top-k value for toploc verification.")

    return parser.parse_args()

def compute_logprob_topk(logits, topk: int = TOPK, reference_output_ids: torch.Tensor | None = None):
    logprob = logits.log_softmax(dim=-1)
    logprob_topk = torch.topk(logprob, k=topk, dim=-1, sorted=True)
    result = {}
    if reference_output_ids is not None:
        result[f'chosen_logprob'] = torch.gather(logprob, 1, torch.tensor(reference_output_ids, device=logprob.device).unsqueeze(-1)).squeeze(-1).tolist()
    for i in range(topk):
        result[f'validator_{i+1}_logprob_output_ids'] = logprob_topk.indices[:, i].tolist()
        result[f'validator_{i+1}_logprob'] = logprob_topk.values[:, i].tolist()
    return result

def compute_gumbel_topk(
    logits,
    generator: torch.Generator,
    reference_noise: torch.Tensor | None = None,
    reference_output_ids: torch.Tensor | None = None,
    topk: int = TOPK,
):
    neg_gumbel_noise = generate_neg_gumbel_noise((logits.shape[0], logits.shape[-1]), generator, logits.device)

    # Make sure the noise we generated matches the noise obtained by the generation sampler
    # This is a sanity check to make sure we are reproducing the noise correctly
    if reference_noise is not None:
        assert reference_output_ids is not None, "reference_output_ids must be provided if reference_noise is provided"
        for i, noise in enumerate(reference_noise):
            assert noise == neg_gumbel_noise[i][reference_output_ids[i]].item()

    variable = logits - neg_gumbel_noise
    variable_topk = torch.topk(variable, k=topk, dim=-1, sorted=True)
    result = {}
    for i in range(topk):
        result[f'validator_{i+1}_output_ids'] = variable_topk.indices[:, i].tolist()
        result[f'validator_{i+1}_variable'] = variable_topk.values[:, i].tolist()
    return result

def get_logits(driver_worker, chunk_states: torch.Tensor):
    model = driver_worker.model_runner.model
    chunk_states = chunk_states.to(model.lm_head.weight.device)
    chunk_logits = model.logits_processor._get_logits(chunk_states, model.lm_head, None)
    return chunk_logits

def main(args):
    if args.attn != "flash":
        raise NotImplementedError("Only flash attention is supported for now.")
    save_dir = Path(args.save_dir)
    # Construct results path based on decode_kv_cache_dtype
    if args.decode_kv_cache_dtype == "auto":
        results_path = save_dir / f'results_{args.decode_model_name.replace("/", "--")}.pt'
    else:
        results_path = save_dir / f'results_{args.decode_model_name.replace("/", "--")}_kvcache_{args.decode_kv_cache_dtype}.pt'
    global results
    global processed_count
    if results is None:
        results = torch.load(results_path)

    llm = LLM(
        model=args.validate_model_name,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_decode_tokens + 1000,
        dtype=args.dtype,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )
    model_executor: MultiprocessingDistributedExecutor = llm.llm_engine.model_executor
    model = model_executor.driver_worker.model_runner.model

    saved_inputs_prefill = []
    def prefill_hook(module, input, output):
        assert isinstance(input[1], torch.Tensor)
        saved_inputs_prefill.append(input[1].clone())
    prefill_hook_handle = model.logits_processor.register_forward_hook(prefill_hook)

    # TODO: Check that there is enough memory left for the topks

    for i, result in tqdm(enumerate(results), total=len(results)):
        if 'validator_logits' in result:
            continue

        tokens_prompt = TokensPrompt(prompt_token_ids=[*result['input_ids'], *result['output_ids'][:-1]])
        sampling_params = SamplingParams(
            temperature=args.temperature,
            #top_p=0.95,
            max_tokens=1,
            logprobs=1,
        )
        saved_inputs_prefill = []
        _ = llm.generate(tokens_prompt, sampling_params, use_tqdm=False)
        assert len(saved_inputs_prefill) == 1

        # Process hidden states in chunks to reduce memory usage
        hidden_states = saved_inputs_prefill[0][len(result['input_ids']) - 1:]
        num_tokens = hidden_states.shape[0]
        g_cuda = torch.Generator(device="cuda").manual_seed(result['seed'])

        #toploc1_results = verify_proofs_bytes(hidden_states[1:], result['toploc1_proof'], decode_batching_size=32, topk=args.toploc_topk, skip_prefill=True)
        #result['toploc1_exp_mismatches'] = [i.exp_mismatches for i in toploc1_results]
        #result['mant_err_means'] = [i.mant_err_mean for i in toploc1_results]
        #result['mant_err_medians'] = [i.mant_err_median for i in toploc1_results]

        for chunk_start in range(0, num_tokens, MAX_CHUNK_SIZE):
            chunk_end = min(chunk_start + MAX_CHUNK_SIZE, num_tokens)
            chunk_hidden_states = hidden_states[chunk_start:chunk_end]

            if args.tp > 1:
                exec_results = model_executor._run_workers(get_logits, chunk_hidden_states, async_run_tensor_parallel_workers_only=False)
                chunk_logits = exec_results[0]
            else:
                chunk_logits: torch.Tensor = model.logits_processor._get_logits(chunk_hidden_states, model.lm_head, None)

            chunk_logits /= args.temperature
            chunk_result = {}
            # This is a hack around a possibility of a deadlock that I cant find
            start_time = time.time()
            while time.time() - start_time < 5:
                if torch.cuda.current_stream().query():
                    break
                time.sleep(0.001)
            if not torch.cuda.current_stream().query():
                print("Deadlock detected, exiting")
                raise KeyboardInterrupt
            chunk_result['validator_logits'] = torch.gather(chunk_logits, 1, torch.tensor(result['output_ids'][chunk_start:chunk_end], device=chunk_logits.device).unsqueeze(-1)).squeeze(-1).tolist()
            chunk_result.update(compute_logprob_topk(chunk_logits, reference_output_ids=torch.tensor(result['output_ids'][chunk_start:chunk_end], device=chunk_logits.device, dtype=torch.long)))
            #chunk_result.update(compute_gumbel_topk(chunk_logits, g_cuda, result['noise_at_output_ids'][chunk_start:chunk_end], result['output_ids'][chunk_start:chunk_end]))
            chunk_result.update(compute_gumbel_topk(chunk_logits, g_cuda, None, result['output_ids'][chunk_start:chunk_end]))
            if chunk_start == 0:
                result.update(chunk_result)
            else:
                for k, v in chunk_result.items():
                    result[k].extend(v)

            # Free memory for the next fwd
            del chunk_logits
            torch.cuda.empty_cache()

        processed_count += 1
        if args.debug:
            print("Naive argmax matchrate", sum([i == j for i, j in zip(result['output_ids'], result['validator_1_logprob_output_ids'])]), "/", len(result['output_ids']))
            print("Naive 2nd matchrate", sum([i == j for i, j in zip(result['output_ids'], result['validator_2_logprob_output_ids'])]), "/", len(result['output_ids']))
            print("Naive 3rd matchrate", sum([i == j for i, j in zip(result['output_ids'], result['validator_3_logprob_output_ids'])]), "/", len(result['output_ids']))
            print("Gumbel argmax matchrate", sum([i == j for i, j in zip(result['output_ids'], result['validator_1_output_ids'])]), "/", len(result['output_ids']))
            print("Gumbel 2nd matchrate", sum([i == j for i, j in zip(result['output_ids'], result['validator_2_output_ids'])]), "/", len(result['output_ids']))
            print("Gumbel 3rd matchrate", sum([i == j for i, j in zip(result['output_ids'], result['validator_3_output_ids'])]), "/", len(result['output_ids']))


    df = pd.DataFrame(results)

    assert 'input_ids' in df.columns
    assert 'output_ids' in df.columns
    assert 'logits' in df.columns
    assert 'noise_at_output_ids' in df.columns
    assert 'seed' in df.columns
    assert 'validator_logits' in df.columns
    for i in range(1, TOPK + 1):
        assert f'validator_{i}_output_ids' in df.columns
        assert f'validator_{i}_variable' in df.columns
        assert f'validator_{i}_logprob' in df.columns
        assert f'validator_{i}_logprob_output_ids' in df.columns

    # Include decode_kv_cache_dtype in the output filename
    if args.decode_kv_cache_dtype == "auto":
        output_file = save_dir / f'validation_{args.validate_model_name.replace("/", "--")}_{args.attn}_{args.dtype}_{args.tp}_{args.device_str}_on_{args.decode_model_name.replace("/", "--")}.parquet'
    else:
        output_file = save_dir / f'validation_{args.validate_model_name.replace("/", "--")}_{args.attn}_{args.dtype}_{args.tp}_{args.device_str}_on_{args.decode_model_name.replace("/", "--")}_kvcache_{args.decode_kv_cache_dtype}.parquet'
    print(f"Saving to {output_file}")
    df.to_parquet(output_file, index=False)
    print(df.info())

    del llm

if __name__ == "__main__":
    args = parse_args()
    save_dir = Path(args.save_dir)
    if args.decode_kv_cache_dtype == "auto":
        temp_file = save_dir / f'temp_validation_{args.validate_model_name.replace("/", "--")}_{args.attn}_{args.dtype}_{args.tp}_{args.device_str}_on_{args.decode_model_name.replace("/", "--")}.pt'
    else:
        temp_file = save_dir / f'temp_validation_{args.validate_model_name.replace("/", "--")}_{args.attn}_{args.dtype}_{args.tp}_{args.device_str}_on_{args.decode_model_name.replace("/", "--")}_kvcache_{args.decode_kv_cache_dtype}.pt'
    try:
        if temp_file.exists():
            results = torch.load(temp_file)
            with open(temp_file.with_suffix(".txt"), "r") as f:
                processed_count = int(f.read())
            print(f"Resuming from {processed_count}")
        main(args)
    except KeyboardInterrupt:
        print(f"Saving to {temp_file}")
        torch.save(results, temp_file)
        with open(temp_file.with_suffix(".txt"), "w") as f:
            f.write(str(processed_count))
