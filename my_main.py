
import torch
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from datasets import load_dataset

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_batch, speculative_sampling_v2
from globals import Decoder
from human_eval.data import read_problems


# my local models
MODELZOO = {
    # llama-1
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "llama1b": "/share_nfs/fangjiarui/root/code/hf_models/TinyLlama-1.1B-step-50K-105b",
    "llama7b": "/share_nfs/tianzhi/code/llama-7b",
    "llama30b": "/share_nfs/fangjiarui/root/code/hf_models/llama-30b-hf",
    "llama2-7b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-7b-hf",
    "llama2-70b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-70b-hf",
    "bloom-560m": "bigscience/bloom-560m",
    "bloom-7b": "bigscience/bloom-7b1",
    "baichuan-7b": "baichuan-inc/Baichuan-7B",
    "baichuan-13b": "baichuan-inc/Baichuan-13B",
    "wizardcoder-7b": "WizardLM/WizardCoder-Python-7B-V1.0",
    "wizardcoder-13b": "WizardLM/WizardCoder-Python-13B-V1.0",
    "wizardcoder-34b": "WizardLM/WizardCoder-Python-34B-V1.0", 
    "starcoder-1b": "WizardLM/WizardCoder-1B-V1.0",
    "starcoder-3b": "WizardLM/WizardCoder-3B-V1.0",
    "starcoder-15b": "WizardLM/WizardCoder-15B-V1.0",
    "codegen-350M": "Salesforce/codegen-350M-mono",
    "codegen-2B": "Salesforce/codegen-2B-mono",
    "codegen-6B": "Salesforce/codegen-6B-mono",
    "codegen-16B": "Salesforce/codegen-16B-mono"
}
prompt_he_0 = "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
prompt_he_5 = "from typing import List\n\n\ndef intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\" Insert a number 'delimeter' between every two consecutive elements of input list `numbers'\n    >>> intersperse([], 4)\n    []\n    >>> intersperse([1, 2, 3], 4)\n    [1, 4, 2, 4, 3]\n    \"\"\"\n"
prompt_natural = "A quick fox jumped "

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    parser.add_argument('--input', type=str, default=prompt_natural)
    parser.add_argument('--approx_model_name', type=str, default=MODELZOO["wizardcoder-7b"])
    parser.add_argument('--target_model_name', type=str, default=MODELZOO["wizardcoder-34b"])
    parser.add_argument('--verbose', '-v', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=300, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    args = parser.parse_args()
    return args

fixed_starter = "Here's the Python script for the given problem:\n\n\n```python\n"

def alpaca_prompt(input):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input}

### Response:"""
    return INSTRUCTION

def color_print(text):
    print(Fore.RED + text + Style.RESET_ALL)
    
def benchmark(fn, print_prefix, use_profiler=True, *args, **kwargs):
    TEST_TIME = 10
    profile_filename = f"./profile_logs/{print_prefix}"
    
    with contexttimer.Timer() as t:
        if use_profiler:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_filename),
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as prof:
                for _ in range(TEST_TIME): 
                    output = fn(*args, **kwargs)
                    prof.step()
        else:
            for _ in range(TEST_TIME): 
                output = fn(*args, **kwargs)

    print(f"\n [benchmark] {print_prefix}, tokens/sec: {len(output[0]) / t.elapsed / TEST_TIME}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")

def generate(input_text, approx_model_name, target_model_name, num_tokens=20, gamma = 4,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False,
             stopping_logits = []):
    # Prepare input
    # APPS
    all_dict = load_dataset("codeparrot/apps", split="test")
    all_question_nums = [4087, 4142, 4148]
    # # HumanEval
    # all_dict = read_problems()
    # all_question_nums = [7, 13, 18, 14, 15]
    
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True, padding_side='left')
    Decoder().set_tokenizer(tokenizer)
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True,
                                                       use_cache=True,
                                                    #    load_in_8bit=False
                  )
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True,
                                                       use_cache=True,
                                                    #    load_in_8bit=False
                  )
    small_model.eval()
    large_model.eval()
    print("finish loading models")
    
    # for question_num in all_question_nums:
    #     prompt = all_dict[f"HumanEval/{question_num}"]["prompt"]
    for question in all_dict:
        number = question["problem_id"]
        if number not in all_question_nums:
            continue
        prompt = question["question"]
        prompt = prompt.replace('    ', '\t')
        input_text = [alpaca_prompt(prompt)]
        input_ids = tokenizer.batch_encode_plus(
                    input_text, 
                    return_tensors="pt",
                    padding=True
                    # truncation=True,
                    # max_length=2048
                ).to(torch.cuda.current_device())['input_ids']
        top_k = 20
        top_p = 0.95
        temperature = 0.01


        # Target model
        print(f"Generating with target model...")
        start = time.time()
        torch.manual_seed(123)
        output = autoregressive_sampling(
            input_ids, 
            large_model, 
            num_tokens, 
            top_k=top_k, 
            top_p=top_p,
            temperature=temperature,
            stopping_logits=stopping_logits
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_text = generated_text[len(input_text[0]):]
        color_print(f"{generated_text}")
        if use_benchmark:
            benchmark(autoregressive_sampling, "AS_large", use_profiling,
                      input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
        print(f"target model time: {time.time() - start}")
        torch.cuda.empty_cache()

        # # Draft model
        # print(f"Generating with draft model...")
        # start = time.time()
        # torch.manual_seed(123)
        # input_ids = input_ids.to(small_model.device)
        # output = autoregressive_sampling(
        #     input_ids,
        #     small_model,
        #     num_tokens,
        #     top_k=top_k,
        #     top_p=top_p,
        #     temperature=temperature,
        #     stopping_logits=stopping_logits
        # )
        # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)    
        # generated_text = generated_text[len(input_text):]
        # color_print(f"{generated_text}")
        # if use_benchmark:
        #     benchmark(autoregressive_sampling, "AS_small", use_profiling,
        #               input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
        # print(f"draft model time: {time.time() - start}")
        
        # # Batched Google Speculative Decoding
        # print(f"Generating with google's speculative_sampling...")
        # start = time.time()
        # torch.manual_seed(123)
        # output = speculative_sampling_batch(
        #     input_ids, 
        #     small_model, 
        #     large_model, 
        #     num_tokens, 
        #     top_k=top_k, 
        #     top_p=top_p, 
        #     gamma=args.gamma,
        #     random_seed=random_seed,
        #     temperature=temperature,
        #     verbose=verbose,
        #     stopping_logits = stopping_logits,
        #     eos_token_id = tokenizer.eos_token_id
        # )
        # generated_batch = [tokenizer.decode(output[i][len(input_ids[i]):], skip_special_tokens=True) for i in range(len(output))]
        # for generated_text in generated_batch:
        #     color_print(f"{generated_text}")
        # print(f"google's batched speculative_sampling time: {time.time() - start}")
        # torch.cuda.empty_cache()

        # # Google Speculative Decoding
        # print(f"Generating with google's speculative_sampling...")
        # start = time.time()
        # torch.manual_seed(123)
        # output = speculative_sampling(
        #     input_ids, 
        #     small_model, 
        #     large_model, 
        #     num_tokens, 
        #     top_k=top_k, 
        #     top_p=top_p, 
        #     gamma = args.gamma,
        #     random_seed=random_seed,
        #     temperature=temperature,
        #     verbose=verbose,
        #     stopping_logits = stopping_logits,
        #     eos_token_id = tokenizer.eos_token_id
        # )
        # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # generated_text = generated_text[len(input_text):]
        # color_print(f"{generated_text}")
        # print(f"google's speculative_sampling time: {time.time() - start}")
        # torch.cuda.empty_cache()

        # # Deepmind Speculative Decoding
        # print(f"Generating with deepmind's speculative_sampling...")
        # start = time.time()
        # torch.manual_seed(123)
        # output = speculative_sampling_v2(
        #     input_ids, 
        #     small_model, 
        #     large_model, 
        #     num_tokens, 
        #     top_k=top_k, 
        #     top_p=top_p, 
        #     gamma = args.gamma,
        #     random_seed=random_seed,
        #     temperature=temperature,
        #     verbose=verbose,
        #     stopping_logits = stopping_logits
        # )
        # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # generated_text = generated_text[len(input_text):]
        # color_print(f"{generated_text}")
        # print(f"deepmind's speculative_sampling time: {time.time() - start}")
        # torch.cuda.empty_cache()



if __name__ == "__main__":
    args = parse_arguments()
    print(f"args.input: {args.input}")
    print(f"args.approx_model_name: {args.approx_model_name}")
    print(f"args.target_model_name: {args.target_model_name}")
    print(f"args.max_tokens: {args.max_tokens}")
    print(f"args.gamma: {args.gamma}")
    print(f"args.seed: {args.seed}")
    print(f"args.verbose: {args.verbose}")
    print(f"args.benchmark: {args.benchmark}")
    stopping_logits = []
    if "Wizard" in args.target_model_name:
        stopping_logits = [[13,28956,13], [13,28956,30004], [13,30004,13,2158]]
        # [13,29937], 
    generate(
        args.input, 
        args.approx_model_name, 
        args.target_model_name, 
        num_tokens=args.max_tokens,
        gamma=args.gamma,
        random_seed = args.seed, 
        verbose=args.verbose, 
        use_benchmark = args.benchmark,
        stopping_logits = stopping_logits
    )
