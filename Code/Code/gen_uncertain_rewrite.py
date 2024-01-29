
import os
import openai
import json

openai.api_key='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'

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

debug=False


def GPT_self_prompt(prompt_str, content_to_be_detected, prefix):

    # import pdb; pdb.set_trace()

    response = openai_backoff(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": f"{prompt_str}: \"{content_to_be_detected}\" {prefix}",
                        }
                    ],
                )
    spit_out = response["choices"][0]["message"]["content"].strip()
    print(spit_out)
    return spit_out

prompt_list = ['Refine the code for me please'] # invariance
prompt_list = prompt_list * 10
prefix='. No need to explain. Just write code:'

with open(f'code_human-v2.json', 'r') as file:
    human = json.load(file)

with open(f'code_GPT-v2.json', 'r') as file:
    GPT = json.load(file)


def rewrite_json(input_json, prompt_list, human=False):
    all_data = []
    for cc, data in enumerate(input_json):
        tmp_dict ={}
        if human:
            tmp_dict['input'] = data[0] + data[1] # prmpot + solution, should be one, but saved separately.
        else:
            tmp_dict['input'] = data[1]  # this is the GPT rewritten, which already contain prompt and solution

        for rept_id, ep in enumerate(prompt_list):
            tmp_dict[str(rept_id) + '_' + ep] = GPT_self_prompt(ep, tmp_dict['input'], prefix)
        
        all_data.append(tmp_dict)

        if debug:
            break
    return all_data

human_rewrite = rewrite_json(human, prompt_list, True)
with open(f'uncertain_rewrite_code_human.json', 'w') as file:
    json.dump(human_rewrite, file, indent=4)

GPT_rewrite = rewrite_json(GPT, prompt_list)
with open(f'uncertain_rewrite_code_GPT.json', 'w') as file:
    json.dump(GPT_rewrite, file, indent=4)





