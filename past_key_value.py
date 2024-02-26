from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LogitsProcessor, LogitsProcessorList
import time
import json
import os
import argparse
import torch
from human_eval.data import write_jsonl, read_problems, stream_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, default=2, help="Model name")
parser.add_argument("--pass_at", type=int, default=1, help="pass @ how many")
FLAGS = parser.parse_args()

# We will hard-code the stop tokens for llama code family, as the tokenizer is automatically adding start tokens
# stop_words = ["\n\n", ("\n","\n"), "\r\n\r\n"]
# stop_words_ids = [[13,13],[30004,13,30004,13]]
stop_words = ["\n#", "\n```\n"]
stop_words_ids = [[13,29937], [13,28956,13], [13,28956,30004]]
assert_stop_words = ["assert"] + stop_words
assert_stop_words_ids = [[9294]] + stop_words_ids
eos_id = 2
eos_token = "</s>"
imports = "\nimport math\nfrom typing import List\n"

def trim_substring_from_end(answer, b):
    while answer.endswith(b):
        answer = answer[:-len(b)]
    return answer

def trim_answer_from_start(answer):
    # Remove all beginning lines in answer, till it starts with "def ", "from" or "import"
    lines = answer.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("def ") or line.startswith("from") or line.startswith("import"):
            break
    answer = "\n".join(lines[i:])
    return answer

def process_answer(answer):
    answer = answer.replace("\r", "")
    answer = answer.replace("\t", "    ")
    answer = trim_answer_from_start(answer)
    answer = trim_substring_from_end(answer, "\n```\n")
    answer = trim_substring_from_end(answer, eos_token)
    answer = trim_substring_from_end(answer, "#")
    answer = trim_substring_from_end(answer, "```")
    answer = trim_substring_from_end(answer, "\n\n")
    return answer

def alpaca_prompt(input):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input}

### Response:"""
    return INSTRUCTION


def main(args):
    loading_start = time.time()
    number_key = "task_id"
    prompt_key = "prompt"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pass_at = args.pass_at
    
    # Load HumanEval Dataset
    all_questions_dict = read_problems()
    all_keys = ["HumanEval/0", "HumanEval/8", "HumanEval/145"]

    # Prepare the model checkpoint
    answer_dict_list = []
    counter = 0
    if args.model == 0:
        model_size = "1B"
        checkpoint = "WizardLM/WizardCoder-1B-V1.0"
    elif args.model == 1:
        model_size = "3B"
        checkpoint = "WizardLM/WizardCoder-3B-V1.0"
    elif args.model == 2:
        model_size = "7B"
        checkpoint = f"WizardLM/WizardCoder-Python-7B-V1.0"
    elif args.model == 3:
        model_size = "13B"
        checkpoint = f"WizardLM/WizardCoder-Python-13B-V1.0"
    elif args.model == 4:
        model_size = "15B"
        checkpoint = "WizardLM/WizardCoder-15B-V1.0"
    elif args.model == 5:
        model_size = "34B"
        checkpoint = f"WizardLM/WizardCoder-Python-34B-V1.0"
    print(f"Model is {checkpoint}")
    print(f"Pass @ {args.pass_at}")
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    loading_end = time.time()
    print(f"Time to load model is {loading_end - loading_start}")
    
    # Stopping criteria for generation using the LogitsProcessor class
    class StopSequences(LogitsProcessor):
        def __init__(self, stop_ids, batch_size, encounters=1, eos_token_id=2):
            StoppingCriteria.__init__(self)
            self.stop_sequences = stop_ids
            self.batch_size = batch_size
            self.encounters = [encounters] * batch_size
            self.NUM_ENCOUNTERS = encounters
            self.eos_token_id = eos_token_id

        def __call__(self, input_ids, scores):
            forced_eos = torch.full((scores.size(1),), -float("inf"))
            forced_eos[self.eos_token_id] = 0
            for stop in self.stop_sequences:
                # Check if the input_ids end with the stop sequence
                for i in range(self.batch_size):
                    if self.encounters[i] <= 0:
                        continue
                    if input_ids[i][-len(stop):].tolist() == stop:
                        self.encounters[i] -= 1
                        if self.encounters[i] <= 0:
                            scores[i] = forced_eos
            return scores

    # Go through each question
    for question_key in all_keys:
        question = all_questions_dict[question_key]
        number = int(question[number_key].split("/")[1])
        print(f"On question {number}")
        prompt = question[prompt_key]
        print(prompt)
        prompt = prompt.replace('    ', '\t')
        prompt = alpaca_prompt(prompt)
        prompt_ids = tokenizer.batch_encode_plus([prompt]*max(pass_at,1), return_tensors="pt", truncation=True, max_length=2048).to(torch.cuda.current_device())
        logits_processor = LogitsProcessorList([StopSequences(stop_words_ids, batch_size=max(pass_at,1), encounters=1)])
        print(f"prompt ids has length {prompt_ids['input_ids'].shape}")
        
        # Generate answers
        start = time.time()
        max_new_tokens = 64
        with torch.no_grad():
            if pass_at in [0,1]:
                output = model(
                    **prompt_ids,
                    use_cache = True,
                    output_attentions = False,
                    output_hidden_states = True,
                )
            else:
                output = model(
                    **prompt_ids,
                    use_cache = True,
                    pad_token_id = tokenizer.eos_token_id,
                    eos_token_id = tokenizer.eos_token_id,
                    max_new_tokens = max_new_tokens,
                    do_sample = True,
                    top_k = 0,
                    top_p = 0.95,
                    temperature = 0.8,
                    num_beams = 1,
                    logits_processor = logits_processor
                )
        # past kv has shape [32, 2, 32, [len prompt], 128])
        
        print(output.keys())
        # print(output.past_key_values)
        print(f"{output.past_key_values[0][0].shape}")
        print(f"{output.past_key_values[0][1].shape}")
        print(f"prompt ids has length {prompt_ids['input_ids'].shape}")
        print(f"past_key_values has length {len(output.past_key_values)}")
        print(f"past_key_values[0] has length {len(output.past_key_values[0])}; "
              f"past_key_values[1] has length {len(output.past_key_values[1])}; "
              f"past_key_values[2] has length {len(output.past_key_values[2])}.")
        print(len(output.attentions))
        input()
        
                

if __name__== "__main__":
    main(FLAGS)
