

import os
import numpy as np
import torch
import json

from transformers import AutoTokenizer,AutoModelForCausalLM
model_path = "/proj/vondrick3/bigmodels/llama2_chat/converted_weights_llama_chat7b"
modeltype = 'llama2_7b_chat'

model_path = "/proj/vondrick3/bigmodels/llama2_chat/converted_weights_llama_chat70b"
modeltype = 'llama2_70b_chat'


def tokenize_and_normalize(sentence):
    # Tokenization and normalization
    return [word.lower().strip() for word in sentence.split()]


tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")


with open(f'../yelp_subset.json', 'r') as file:
    yelp = json.load(file)


GPT = []
cnt = 0
cutoff=400
debug = False
save_interval=20
for each in yelp:
    prompt_str = each['text']
    prompts = f"Write a very short and concise review based on this: {prompt_str}"

    # prompts='a fat mushroom is'

    #prompts either a string or a list of strings
    model_inputs = tokenizer(prompts, return_tensors="pt").to("cuda:0")
    model_inputs.pop("token_type_ids", None)

    output = model.generate(**model_inputs, max_new_tokens=len(tokenize_and_normalize(prompts)))

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    print('length', len(tokenize_and_normalize(prompts)), len(prompts))
    print(decoded_output)

    GPT.append(decoded_output)
    cnt += 1
    if cnt == cutoff:
        break

    if debug:
        break

    if cnt % save_interval==0:
        with open(f'{modeltype}_yelp_concise_400test_only.json', 'w') as file:
            json.dump(GPT, file, indent=4)

# print('GPT', GPT)
with open(f'{modeltype}_yelp_concise_400test_only.json', 'w') as file:
    json.dump(GPT, file, indent=4)



