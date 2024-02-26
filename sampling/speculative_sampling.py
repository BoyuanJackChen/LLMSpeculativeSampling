import torch
from tqdm import tqdm
import torch
from typing import List

from sampling.kvcache_model import KVCacheModel
from sampling.utils import norm_logits, sample, max_fn
from globals import Decoder

def check_prefix_ending(prefix, stopping_logits, t_len):
    for stop_tensor in stopping_logits:
        for i in range(t_len):
            window = prefix[0][-i-len(stop_tensor):]
            window = window[:len(stop_tensor)]
            if torch.equal(window, stop_tensor):
                # print("aha!")
                return True
    return False

@torch.no_grad()
def speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None,
                         stopping_logits : List[int] = []) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        prefix (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    stopping_logits = [torch.tensor(x).to(prefix.device) for x in stopping_logits]
    T = seq_len + max_len

    assert prefix.shape[0] == 1, "input batch size must be 1"
    # assert approx_model.device == target_model.device
    
    # Devices for each model
    approx_device = approx_model.device
    target_device = target_model.device
    prefix = prefix.to(approx_device)
    
    # We call .generate using these cached models
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]
        x_approx_device = approx_model_cache.generate(prefix, gamma)
        
        # Move generated tokens to target_model device for checking
        x_target_device = x_approx_device.to(target_device)
        _ = target_model_cache.generate(x_target_device, 1)
        # _ = target_model_cache.generate(x, 1)
        
        n = prefix_len + gamma - 1

        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = target_device)
            j_approx_device = x_approx_device[:, prefix_len + i]
            j_target_device = x_target_device[:, prefix_len + i]
            
            # For each guessed word, accept if target model prob / approx model prob < r. The probability is not normalized
            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j_target_device]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j_approx_device]).to(target_device):
                # reject
                n = prefix_len + i - 1
                break
            
            if verbose:
                print(f"approx guess accepted {j_approx_device[0]}: \033[31m{Decoder().decode(torch.tensor([j_approx_device]))}\033[0m")

            accepted_count += 1
        
        # n is the index of the last accepted token
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x_approx_device[:, :n + 1]
        # print(f"n is: {n}; prefix has length {prefix_len}, is: {prefix[0]}")
        # print(f"Now calling rollback...")
        approx_model_cache.rollback(n+1)
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n. Prob dist is the difference between target and approx models
            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :].to(target_device)))
            if verbose:
                print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            # t is the token sampled from the probability distribution, on the last word
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n+2)
        
        prefix = torch.cat((prefix, t.to(approx_device)), dim=1)
        if len(stopping_logits)>0 and check_prefix_ending(prefix, stopping_logits, n-prefix_len+2):
            break

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
    return prefix



@torch.no_grad()
def speculative_sampling_v2(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None, verbose : bool = False,
                         stopping_logits : List[int] = []) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    resample_count = 0
    target_sample_count = 0
    accepted_count = 0

    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], 
                                  temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = target_model(x).logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            
            is_all_accept = True
            n = prefix_len - 1
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = p.device)
                j = x[:, prefix_len + i]
                
                # For each guessed word, accept if target model prob / approx model prob > r. The probability is normalized
                if r < torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]):
                    # accept, and update n
                    n += 1
                    accepted_count += 1
                else:
                    # reject
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    is_all_accept = False
                    resample_count += 1
                    break
         
            prefix = x[:, :n + 1]
            
            if is_all_accept:
                t = sample(p[:, -1, :])
                target_sample_count += 1
            
            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(n - pbar.n)
            if len(stopping_logits)>0 and check_prefix_ending(prefix, stopping_logits, n-prefix_len+2):
                break

    if verbose: 
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")

    return prefix

