
"""Using LLaMa to rewrite for cheap detection """
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


import os
import openai
import json

debug = False
def GPT_self_prompt(prompt_str, content_to_be_detected):

    # import pdb; pdb.set_trace()

    # response = openai_backoff(
    #                 model="gpt-3.5-turbo",
    #                 messages=[
    #                     {
    #                         "role": "user",
    #                         "content": f"{prompt_str}: \"{content_to_be_detected}\"",
    #                     }
    #                 ],
    #             )
    # spit_out = response["choices"][0]["message"]["content"].strip()
    prompts = f"{prompt_str}: \"{content_to_be_detected}\""
    model_inputs = tokenizer(prompts, return_tensors="pt").to("cuda:0")
    model_inputs.pop("token_type_ids", None)

    output = model.generate(**model_inputs, max_new_tokens=len(tokenize_and_normalize(prompts)))

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    print('length', len(tokenize_and_normalize(prompts)), len(prompts))
    print(decoded_output)

    return decoded_output

prompt_list = ['Revise this with your best effort', 'Help me polish this', 'Rewrite this for me', 
                'Make this fluent while doing minimal change', 'Refine this for me please', 'Concise this for me and keep all the information',
                'Improve this in GPT way']

with open(f'arXiv_human.json', 'r') as file:
    human = json.load(file)

with open(f'arxiv_GPT_concise.json', 'r') as file:
    GPT = json.load(file)


def rewrite_json(input_json, prompt_list, human=False):
    all_data = []
    for cc, data in enumerate(input_json):
        tmp_dict ={}
        
        tmp_dict['input'] = data['abs']

        for ep in prompt_list:
            tmp_dict[ep] = GPT_self_prompt(ep, tmp_dict['input'])
        
        all_data.append(tmp_dict)

        if debug:
            break
    return all_data

human_rewrite = rewrite_json(human, prompt_list, True)
with open(f'llama_rewrite_arxiv_human_inv.json', 'w') as file:
    json.dump(human_rewrite, file, indent=4)

GPT_rewrite = rewrite_json(GPT, prompt_list)
with open(f'llama_rewrite_arxiv_GPT_inv.json', 'w') as file:
    json.dump(GPT_rewrite, file, indent=4)





