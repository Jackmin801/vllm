import asyncio
from tqdm.asyncio import tqdm
import torch
import openai
import argparse
from datasets import load_dataset
from pathlib import Path
import random
from transformers import AutoTokenizer
from typing import List, Dict

def parse_args():
    parser = argparse.ArgumentParser(description="Run activation saving and inference generation with a language model.")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Name of the model to use.")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples to generate.")
    parser.add_argument("--save_dir", type=str, default="just4", help="Directory to save outputs.")
    parser.add_argument("--max_decode_tokens", type=int, default=100, help="Maximum number of decode tokens.")
    parser.add_argument("--dataset_name", type=str, default="PrimeIntellect/Intellect-2-RL-Dataset", help="Dataset to load.")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="System prompt to prepend to each input.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--debug", action="store_true", default=False)
    
    return parser.parse_args()

async def process_single_prompt(
    client: openai.AsyncOpenAI,
    prompt: str,
    model_name: str,
    system_prompt: str,
    max_decode_tokens: int,
    temperature: float,
    tokenizer: AutoTokenizer,
    debug: bool = False
) -> Dict:
    """Process a single prompt asynchronously."""
    seed = random.randint(0, int(1e12))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        logprobs=True,
        max_tokens=max_decode_tokens,
        seed=seed,
        temperature=temperature,
    )
    
    # outputs = response.choices[0].message.content
    output_token_ids = [int(i.token[len("token_id:"):]) for i in response.choices[0].logprobs.content]
    
    if debug:
        print(output_token_ids)
        print(response.choices[0].message.tool_calls)
    
    input_ids = [int(i) for i in response.choices[0].message.tool_calls[0].function.arguments.split(",")]

    result = {
        "input_ids": input_ids,
        "output_ids": output_token_ids,
        "seed": seed,
        "logits": None,
        "noise_at_output_ids": None,
        "toploc1_proof": None,
    }
    
    if debug:
        print(result)
        print(tokenizer.decode(input_ids))
        print(len(input_ids), len(output_token_ids))
        print("-" * 100)
        print(tokenizer.decode(output_token_ids))
    
    return result

async def main(args):
    ds = load_dataset(args.dataset_name, split="train")
    print(ds)
    if "ultrachat" in args.dataset_name:
        prompts = [i['data'][0] for _, i in zip(range(args.n_samples), ds)]
    elif "synthetic-1" in args.dataset_name:
        prompts = [i['messages'][0]['content'] for _, i in zip(range(args.n_samples), ds)]
    else:
        prompts = [i['prompt'] for _, i in zip(range(args.n_samples), ds)]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_save_path = save_dir / f"results_{args.model_name.replace('/', '--')}.pt"

    client = openai.AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="Bearer sk-proj-1234567890")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create async tasks for all prompts
    tasks = []
    for prompt in prompts:
        task = process_single_prompt(
            client=client,
            prompt=prompt,
            model_name=args.model_name,
            system_prompt=args.system_prompt,
            max_decode_tokens=args.max_decode_tokens,
            temperature=args.temperature,
            tokenizer=tokenizer,
            debug=args.debug
        )
        tasks.append(task)
    
    # Run all tasks concurrently with progress bar
    results: List[Dict] = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing prompts"):
        result = await coro
        results.append(result)

    torch.save(results, results_save_path)
    print(f"Saved results to {results_save_path}")

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
