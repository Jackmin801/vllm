import math
import torch

# No expert bias support
# No activation quantization support
# No TP support

NUM_TOKENS = 2 ** 13 # Should do up to 262144 (2 ^ 18)
EP_RANK = 1
EP_SIZE = 32 # 32 and 48 are what we will probably end up using
LORA_RANK = 16
MAX_LORAS = 8

# GLM-5-FP8 configs
HIDDEN_SIZE = 6144
TOP_K = 8
MOE_INTERMEDIATE_SIZE = 2048
GLOBAL_EXPERTS = 256
LOCAL_EXPERTS = GLOBAL_EXPERTS // EP_SIZE

assert MOE_INTERMEDIATE_SIZE % 128 == 0
assert HIDDEN_SIZE % 128 == 0


torch.set_default_device("cuda")
hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16)
# Some of them will be -1, indicating that the token is not assigned to any expert on this ep rank
topk_ids = torch.randint(0, GLOBAL_EXPERTS, (NUM_TOKENS, TOP_K), dtype=torch.int32)
_expert_map = torch.full((GLOBAL_EXPERTS,), -1, dtype=torch.int32)
for i in range(LOCAL_EXPERTS):
    _expert_map[LOCAL_EXPERTS * EP_RANK + i] = i
topk_ids = _expert_map[topk_ids]

topk_weights = torch.randn(NUM_TOKENS, TOP_K, dtype=torch.float32)
w1 = (torch.randn(LOCAL_EXPERTS, 2, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16) * 0.01).to(torch.float8_e4m3fn)
w2 = (torch.randn(LOCAL_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, dtype=torch.bfloat16) * 0.01).to(torch.float8_e4m3fn)
q1 = torch.randn(LOCAL_EXPERTS, 2, MOE_INTERMEDIATE_SIZE // 128, HIDDEN_SIZE // 128, dtype=torch.float32) / math.sqrt(HIDDEN_SIZE)
q2 = torch.randn(LOCAL_EXPERTS, HIDDEN_SIZE // 128, MOE_INTERMEDIATE_SIZE // 128, dtype=torch.float32) / math.sqrt(MOE_INTERMEDIATE_SIZE)
output = torch.zeros(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16)
activation = "silu" # vLLM flashinfer subs silu with SwiGLU?
tune_max_num_tokens = NUM_TOKENS # This is just to inform the kernel what shapes to tune for

# Lora specific
lora_ids = torch.randint(0, MAX_LORAS, (NUM_TOKENS,), dtype=torch.int32)
w1_lora_a = torch.randn(MAX_LORAS, LOCAL_EXPERTS, 2, LORA_RANK, HIDDEN_SIZE, dtype=torch.bfloat16) / math.sqrt(HIDDEN_SIZE)
w1_lora_b = torch.randn(MAX_LORAS, LOCAL_EXPERTS, 2, MOE_INTERMEDIATE_SIZE, LORA_RANK, dtype=torch.bfloat16) / math.sqrt(LORA_RANK)
w2_lora_a = torch.randn(MAX_LORAS, LOCAL_EXPERTS, LORA_RANK, MOE_INTERMEDIATE_SIZE, dtype=torch.bfloat16) / math.sqrt(MOE_INTERMEDIATE_SIZE)
w2_lora_b = torch.randn(MAX_LORAS, LOCAL_EXPERTS, HIDDEN_SIZE, LORA_RANK, dtype=torch.bfloat16) / math.sqrt(LORA_RANK)

# Workspace
# You are allowed to change the sizes and amount here.
# It's just so vLLM knows the memory requirements at load time.
workspace13 = torch.empty(NUM_TOKENS * TOP_K, 2 * MOE_INTERMEDIATE_SIZE, dtype=torch.bfloat16)
workspace2 = torch.empty(NUM_TOKENS * TOP_K, MOE_INTERMEDIATE_SIZE, dtype=torch.bfloat16)
workspace_reduce = torch.empty(NUM_TOKENS * TOP_K, HIDDEN_SIZE, dtype=torch.bfloat16)


def print_tensor_metadata():
    print("=== Start Inputs ===")
    print(f"hidden_states: {hidden_states.shape}, {hidden_states.dtype}, {hidden_states.stride()}")
    print(f"topk_ids: {topk_ids.shape}, {topk_ids.dtype}, {topk_ids.stride()}")
    print(f"topk_weights: {topk_weights.shape}, {topk_weights.dtype}, {topk_weights.stride()}")
    print(f"w1: {w1.shape}, {w1.dtype}, {w1.stride()}")
    print(f"w2: {w2.shape}, {w2.dtype}, {w2.stride()}")
    print(f"q1: {q1.shape}, {q1.dtype}, {q1.stride()}")
    print(f"q2: {q2.shape}, {q2.dtype}, {q2.stride()}")
    print(f"output: {output.shape}, {output.dtype}, {output.stride()}")
    print(f"lora_ids: {lora_ids.shape}, {lora_ids.dtype}, {lora_ids.stride()}")
    print(f"w1_lora_a: {w1_lora_a.shape}, {w1_lora_a.dtype}, {w1_lora_a.stride()}")
    print(f"w1_lora_b: {w1_lora_b.shape}, {w1_lora_b.dtype}, {w1_lora_b.stride()}")
    print(f"w2_lora_a: {w2_lora_a.shape}, {w2_lora_a.dtype}, {w2_lora_a.stride()}")
    print(f"w2_lora_b: {w2_lora_b.shape}, {w2_lora_b.dtype}, {w2_lora_b.stride()}")
    print(f"activation: {activation}")
    print(f"tune_max_num_tokens: {tune_max_num_tokens}")
    print("=== End Inputs ===")

def dequantize_fp8_block(w_fp8, scales, block_size=128):
    """Dequantize FP8 weights with per-block scales."""
    shape = w_fp8.shape
    *batch, M, K = shape
    w = w_fp8.to(torch.bfloat16).reshape(*batch, M // block_size, block_size,
                                          K // block_size, block_size)
    s = scales.reshape(*batch, M // block_size, 1, K // block_size, 1)
    return (w * s).reshape(shape)


def reference_moe_forward(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    output: torch.Tensor,
    activation: str,
    tune_max_num_tokens: int,
    # Lora specific
    lora_ids: torch.Tensor,
    w1_lora_a: torch.Tensor,
    w1_lora_b: torch.Tensor,
    w2_lora_a: torch.Tensor,
    w2_lora_b: torch.Tensor,
    # Workspace
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    workspace_reduce: torch.Tensor,
):
    assert activation == "silu", "Only silu is supported for now"
    num_experts = w1.shape[0]
    output.zero_()

    for expert_id in range(num_experts):
        # Permute inputs — gather all (token, k) pairs routed to this expert
        mask = topk_ids == expert_id
        token_indices, k_indices = torch.where(mask)
        if len(token_indices) == 0:
            continue

        h = hidden_states[token_indices]              # (N, HIDDEN)
        weights = topk_weights[token_indices, k_indices]  # (N,)
        lids = lora_ids[token_indices]                # (N,)

        # Up and Gate
        # Dequantize expert weights
        w1_e = dequantize_fp8_block(w1[expert_id], q1[expert_id])  # (2, INTER, HIDDEN)
        gate = (h.to(torch.float32) @ w1_e[0].T).to(torch.bfloat16)  # (N, INTER)
        up   = (h.to(torch.float32) @ w1_e[1].T).to(torch.bfloat16)  # (N, INTER)

        # Up and Gate LoRA
        for lid in lids.unique():
            m = lids == lid
            h_l = h[m]
            gate[m] += h_l @ w1_lora_a[lid, expert_id, 0].T @ w1_lora_b[lid, expert_id, 0].T
            up[m]   += h_l @ w1_lora_a[lid, expert_id, 1].T @ w1_lora_b[lid, expert_id, 1].T

        # SiLU (SwiGLU: silu(gate) * up)
        intermediate = torch.nn.functional.silu(gate) * up  # (N, INTER)

        # Down
        w2_e = dequantize_fp8_block(w2[expert_id], q2[expert_id])  # (HIDDEN, INTER)
        down = (intermediate.to(torch.float32) @ w2_e.T).to(torch.bfloat16)  # (N, HIDDEN)

        # Down LoRA
        for lid in lids.unique():
            m = lids == lid
            down[m] += intermediate[m] @ w2_lora_a[lid, expert_id].T @ w2_lora_b[lid, expert_id].T

        # Weighted Reduce — unpermute back into output
        output.index_add_(0, token_indices,
                          (weights.unsqueeze(1) * down).to(output.dtype))

print_tensor_metadata()

print("=== Starting output ===")
print(output)
reference_moe_forward(
    hidden_states,
    topk_ids,
    topk_weights,
    w1,
    w2,
    q1,
    q2,
    output,
    activation,
    tune_max_num_tokens,
    lora_ids,
    w1_lora_a,
    w1_lora_b,
    w2_lora_a,
    w2_lora_b,
    workspace13,
    workspace2,
    workspace_reduce,
)

print("=== Ending output ===")
print(output)


# If all -1 means token didnt route to any of the local experts on this ep rank
mask = topk_ids.sum(dim=1) != -8
print(f"Number of routed tokens: {mask.sum()}")
print(f"Routed ratio: {mask.sum() / NUM_TOKENS}")
print(f"Routed output (should be non-zero): {output[mask]}")
print(hidden_states[mask].mean(), hidden_states[mask].std(), hidden_states[mask].norm(dim=1).mean())
print(output[mask].mean(), output[mask].std(), output[mask].norm(dim=1).mean())
print(f"Unrouted output (should be zero): {output[~mask]}")
