

import os
import numpy as np
import torch
import json

from transformers import AutoTokenizer,AutoModelForCausalLM
model_path = "/proj/vondrick3/bigmodels/llama2_chat/converted_weights_llama_chat7b"
modeltype = 'llama2_7b_chat'

# model_path = "/proj/vondrick3/bigmodels/llama2_chat/converted_weights_llama_chat70b"


def tokenize_and_normalize(sentence):
    # Tokenization and normalization
    return [word.lower().strip() for word in sentence.split()]


tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")


codedata=[]
with open(f'../HumanEval.jsonl', 'r') as file:
    for line in file:
        codedata.append(json.loads(line))


GPT = []
cnt = 0
cutoff=400
debug = False
save_interval=20

for each in codedata:
    prompt_str = each['prompt']
    solution = each['canonical_solution']

    prompt1 = f"Describe what does this code do: {prompt_str}{solution}"

    # prompts='a fat mushroom is'

    #prompts either a string or a list of strings
    model_inputs = tokenizer(prompt1, return_tensors="pt").to("cuda:0")
    model_inputs.pop("token_type_ids", None)

    output = model.generate(**model_inputs, max_new_tokens=len(tokenize_and_normalize(prompt1)))

    ans = tokenizer.decode(output[0], skip_special_tokens=True)

    ###
    prompt2 = f"I want to do this: {ans} Help me write python code start with this {prompt_str}, no explanation, just code:"
    model_inputs = tokenizer(prompt2, return_tensors="pt").to("cuda:0")
    model_inputs.pop("token_type_ids", None)
    output = model.generate(**model_inputs, max_new_tokens=len(tokenize_and_normalize(prompt1)))
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    print('length', len(tokenize_and_normalize(prompt1)), len(prompt1))
    print(decoded_output)

    GPT.append((each['prompt'], decoded_output, ans))
    cnt += 1
    if cnt == cutoff:
        break

    if debug:
        break

    if cnt % save_interval==0:
        with open(f'{modeltype}_code_GPT-v2.json', 'w') as file:
            json.dump(GPT, file, indent=4)

# print('GPT', GPT)
with open(f'{modeltype}_code_GPT-v2.json', 'w') as file:
    json.dump(GPT, file, indent=4)



