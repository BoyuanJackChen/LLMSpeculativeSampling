import torch

from tqdm import tqdm
from sampling.utils import norm_logits, sample
from transformers import AutoTokenizer

def check_prefix_ending(prefix, stopping_logits):
    for stop_tensor in stopping_logits:
        if torch.equal(prefix[0][-len(stop_tensor):], stop_tensor):
            print("aha!")
            return True
    return False

@torch.no_grad()
def autoregressive_sampling(x : torch.Tensor, model : torch.nn.Module, N : int, 
                            temperature : float = 1, top_k : int = 0, top_p : float = 0,
                            stopping_logits : list = []) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardCoder-Python-7B-V1.0")
    n = len(x)
    T = len(x) + N
    past_key_values = None
    stopping_tensors = [torch.tensor(a).to(x.device) for a in stopping_logits]
    
    while n < T:
        # outputs = model(x)
        if past_key_values:
            last_ids = x[:, -1]
            if last_ids.dim() == 1:
                last_ids = torch.unsqueeze(last_ids, 0)
            outputs = model(last_ids, past_key_values = past_key_values, use_cache = True)
        else:
            outputs = model(x)
        last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
        past_key_values = outputs.past_key_values
        idx_next = sample(last_p)
        x = torch.cat((x, idx_next), dim=1)
        n += 1
        # print(f"{tokenizer.decode(x[0])}")
        # print(x[0][-5:])
        # input()
        if check_prefix_ending(x, stopping_tensors):
            break
        
    return x

