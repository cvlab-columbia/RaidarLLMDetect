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


def GPT_self_prompt(prompt_str, content_to_be_detected):

    response = openai_backoff(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": f"{prompt_str}: {content_to_be_detected}",
                        }
                    ],
                )
    spit_out = response["choices"][0]["message"]["content"].strip()
    print(spit_out)
    return spit_out

# prompt_list = ['Revise the code with your best effort', 'Help me polish this code', 'Rewrite the code with GPT style', 'Refine the code for me please', 'Concise the code without change the functionality'] # invariance
# prefix='. No need to explain. Just write code'

prompt_list = [
                ['Rewrite to Expand this', 'Rewrite to Concise this'], 
               ['Write this in the opposite tone', 'Write this in the opposite tone'],
               ['Rewrite this in the opposite meaning', 'Rewrite this in the opposite meaning'],
               ]

with open(f'yelp_human.json', 'r') as file:
    human = json.load(file)

with open(f'yelp_GPT_concise.json', 'r') as file:
    GPT = json.load(file)


tmp=100
def rewrite_json(input_json, prompt_list, human=False):
    all_data = []
    for cc, data in enumerate(input_json):
        print(cc, len(input_json))
        tmp_dict ={}

        tmp_dict['input'] = data

        for ep1, ep2 in prompt_list:
            tmp_output_from_prompt = GPT_self_prompt(ep1, tmp_dict['input'])
            final_output_from_prompt = GPT_self_prompt(ep2, tmp_output_from_prompt)

            tmp_dict['tmp&_' + ep1] = tmp_output_from_prompt
            tmp_dict['final*_' + ep2] = final_output_from_prompt
        
        all_data.append(tmp_dict)

        if debug:
            break
        # if cc == 100:
        #     break
    return all_data

human_rewrite = rewrite_json(human, prompt_list)
with open(f'equi_rewrite_yelp_human_fix-100.json', 'w') as file:
    json.dump(human_rewrite, file, indent=4)

GPT_rewrite = rewrite_json(GPT, prompt_list)
with open(f'equi_rewrite_yelp_GPT_fix-100.json', 'w') as file:
    json.dump(GPT_rewrite, file, indent=4)










