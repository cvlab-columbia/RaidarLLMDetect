import os
import openai
import json


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



debug=False


def GPT_self_prompt(prompt_str, content_to_be_detected):
    prompts = f"{prompt_str}: \"{content_to_be_detected}\""
    model_inputs = tokenizer(prompts, return_tensors="pt").to("cuda:0")
    model_inputs.pop("token_type_ids", None)

    output = model.generate(**model_inputs, max_new_tokens=len(tokenize_and_normalize(prompts)))

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    print('length', len(tokenize_and_normalize(prompts)), len(prompts))
    print(decoded_output)

    return decoded_output


    # response = openai_backoff(
    #                 model="gpt-3.5-turbo",
    #                 messages=[
    #                     {
    #                         "role": "user",
    #                         "content": f"{prompt_str}: {content_to_be_detected}",
    #                     }
    #                 ],
    #             )
    # spit_out = response["choices"][0]["message"]["content"].strip()
    # print(spit_out)
    # return spit_out

# prompt_list = ['Revise the code with your best effort', 'Help me polish this code', 'Rewrite the code with GPT style', 'Refine the code for me please', 'Concise the code without change the functionality'] # invariance
# prefix='. No need to explain. Just write code'

prompt_list = ['Revise this with your best effort', 'Help me polish this', 'Rewrite this for me', 
                'Make this fluent while doing minimal change', 'Refine this for me please', 'Concise this for me and keep all the information',
                'Improve this in GPT way']

with open(f'yelp_human.json', 'r') as file:
    human = json.load(file)

with open(f'yelp_GPT_concise.json', 'r') as file:
    GPT = json.load(file)


def rewrite_json(input_json, prompt_list, human=False):
    all_data = []
    for cc, data in enumerate(input_json):
        print(cc, len(input_json))
        tmp_dict ={}

        tmp_dict['input'] = data

        for ep in prompt_list:
            tmp_dict[ep] = GPT_self_prompt(ep, tmp_dict['input'])
        
        all_data.append(tmp_dict)

        if debug:
            break
    return all_data

human_rewrite = rewrite_json(human, prompt_list)
with open(f'llama_rewrite_yelp_human_inv.json', 'w') as file:
    json.dump(human_rewrite, file, indent=4)

GPT_rewrite = rewrite_json(GPT, prompt_list)
with open(f'llama_rewrite_yelp_GPT_inv.json', 'w') as file:
    json.dump(GPT_rewrite, file, indent=4)





