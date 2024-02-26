
import torch
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2
from globals import Decoder
from human_eval.data import read_problems


# my local models
MODELZOO = {
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

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    parser.add_argument('--approx_model_name', type=str, default=MODELZOO["wizardcoder-7b"])
    parser.add_argument('--target_model_name', type=str, default=MODELZOO["wizardcoder-34b"])
    parser.add_argument('--verbose', '-v', action='store_true', default=True, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=50, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    args = parser.parse_args()
    return args


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

def generate(small_model, large_model,
             input_text, approx_model_name, target_model_name, num_tokens=20, gamma = 4,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False):
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
    Decoder().set_tokenizer(tokenizer)
    
    small_model.eval()
    large_model.eval()
    print("finish loading models")
    
    # Prepare input
    all_dict = read_problems()
    # 7, 13, 18
    question_num = 18
    selected_question = all_dict[f"HumanEval/{question_num}"]
    input_text = selected_question["prompt"]
    # input_text = args.input
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch.cuda.current_device())
    # input_ids = tokenizer.encode(input_text, return_tensors='pt')
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
        temperature=temperature
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text = generated_text[len(input_text):]
    color_print(f"{generated_text}")
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_large", use_profiling,
                  input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    print(f"target model time: {time.time() - start}")
    torch.cuda.empty_cache()


    # Draft model
    print(f"Generating with draft model...")
    start = time.time()
    torch.manual_seed(123)
    input_ids = input_ids.to(small_model.device)
    output = autoregressive_sampling(
        input_ids,
        small_model,
        num_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)    
    generated_text = generated_text[len(input_text):]
    color_print(f"{generated_text}")
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_small", use_profiling,
                  input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    print(f"draft model time: {time.time() - start}")
    
    
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
    #     temperature=temperature
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
    #     temperature=temperature
    # )
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # generated_text = generated_text[len(input_text):]
    # color_print(f"{generated_text}")
    # print(f"deepmind's speculative_sampling time: {time.time() - start}")
    # torch.cuda.empty_cache()



if __name__ == "__main__":
    args = parse_arguments()
    print(f"args.approx_model_name: {args.approx_model_name}")
    print(f"args.target_model_name: {args.target_model_name}")
    print(f"args.max_tokens: {args.max_tokens}")
    print(f"args.gamma: {args.gamma}")
    print(f"args.seed: {args.seed}")
    print(f"args.verbose: {args.verbose}")
    print(f"args.benchmark: {args.benchmark}")
    small_model = AutoModelForCausalLM.from_pretrained(
                        args.approx_model_name, 
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        use_cache=True,
                        load_in_8bit=False
                    )
    large_model = AutoModelForCausalLM.from_pretrained(
                        args.target_model_name, 
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        use_cache=True,
                        load_in_8bit=False
                    )
    
    
    
    generate(
        args.input, 
        args.approx_model_name, 
        args.target_model_name, 
        num_tokens=args.max_tokens,
        gamma=args.gamma,
        random_seed = args.seed, 
        verbose=args.verbose, 
        use_benchmark = args.benchmark
    )
