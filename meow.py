from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.attention.backends.flash_attn import FlashAttentionMetadataBuilder, FlashAttentionMetadata
from vllm.forward_context import set_forward_context
import torch

MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model=MODEL_NAME, tensor_parallel_size=4, max_model_len=4096)
model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
model = llm.llm_engine.model_executor.driver_worker.model_runner.model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(model_runner)
print(dir(model_runner))
#from vllm.config import VllmConfig, set_current_vllm_config
#vllm_config = VllmConfig()

prefill_tensors = []
def foo(module, args, output):
    print("Hello!!")
    prefill_tensors.append(output[0].detach().clone())

handle = model.model.register_forward_hook(foo)

def get_attn_metadata(seq_len: int):
    return FlashAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=seq_len,
        num_decode_tokens=0,
        slot_mapping=torch.arange(0, seq_len, device="cuda:0", dtype=torch.long), #torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda:0'),
        multi_modal_placeholder_index_maps={},
        seq_lens=[seq_len],
        seq_lens_tensor=torch.tensor([seq_len], device='cuda:0', dtype=torch.int32),
        max_prefill_seq_len=seq_len,
        max_decode_seq_len=0,
        context_lens_tensor=torch.tensor([0], device='cuda:0', dtype=torch.int32),
        block_tables=torch.tensor([], device='cuda:0', dtype=torch.int32),
        use_cuda_graph=False,
        max_query_len=seq_len,
        max_decode_query_len=1,
        query_start_loc=torch.tensor([0, seq_len], device='cuda:0', dtype=torch.int32),
        seq_start_loc=torch.tensor([0, seq_len], device='cuda:0', dtype=torch.int32),
        _cached_prefill_metadata=None,
        #_cached_prefill_metadata=FlashAttentionMetadata(
            #num_prefills=1,
            #num_prefill_tokens=8, 
            #um_decode_tokens=0,
            #slot_mapping=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda:0'),
            #multi_modal_placeholder_index_maps={}, 
            #seq_lens=[8], seq_lens_tensor=tensor([8], device='cuda:0', dtype=torch.int32), 
            #max_prefill_seq_len=8, 
            #max_decode_seq_len=0, 
            #context_lens_tensor=tensor([0], device='cuda:0', dtype=torch.int32), 
            #block_tables=tensor([], device='cuda:0', size=(1, 0), dtype=torch.int32), 
            #use_cuda_graph=False, 
            #max_query_len=8, 
            #max_decode_query_len=0, 
            #query_start_loc=tensor([0, 8], device='cuda:0', dtype=torch.int32), 
            #seq_start_loc=tensor([0, 8], device='cuda:0', dtype=torch.int32),
            #_cached_prefill_metadata=None, 
            #_cached_decode_metadata=None, 
            #encoder_seq_lens=None, 
            #encoder_seq_lens_tensor=None, 
            #encoder_seq_start_loc=None, 
            #max_encoder_seq_len=None, 
            #num_encoder_tokens=None, cross_slot_mapping=None, cross_block_tables=None), 
        _cached_decode_metadata=None, 
        encoder_seq_lens=None, 
        encoder_seq_lens_tensor=None, 
        encoder_seq_start_loc=None, 
        max_encoder_seq_len=None, 
        num_encoder_tokens=None, 
        cross_slot_mapping=None, 
        cross_block_tables=None
    )

def get_prefill_and_decode_tensors(input_text: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    prompts = [ input_text ]
    output = llm.generate(prompts, sampling_params)[0]

    print([[*tokenizer(input_text).input_ids, *output.outputs[0].token_ids]])
    token_ids = torch.tensor([[*tokenizer(input_text).input_ids, *output.outputs[0].token_ids]], device='cuda:0')
    print(token_ids.shape, token_ids)

    kv_caches = [torch.zeros(0) for _ in range(80)]
    positions = torch.arange(0, token_ids.shape[1], device=token_ids.device, dtype=torch.long)
    attn_metadata = get_attn_metadata(token_ids.shape[1])
    with set_forward_context(attn_metadata, model_runner.vllm_config):
        with torch.inference_mode():
            _ = model(token_ids, positions, kv_caches, attn_metadata)
    
    print("Prefill:", [i.shape for i in prefill_tensors])

prompts = [
    "What is the capital of France?",
    "What is the capital of Germany?",
    #"What is the capital of Italy?",
    #"What is the capital of Spain?",
]

for prompt in prompts:
    get_prefill_and_decode_tensors(prompt)
