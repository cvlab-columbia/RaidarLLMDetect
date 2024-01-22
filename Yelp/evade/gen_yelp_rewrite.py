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

    # import pdb; pdb.set_trace()

    response = openai_backoff(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": f"{prompt_str}: \"{content_to_be_detected}\"",
                        }
                    ],
                )
    spit_out = response["choices"][0]["message"]["content"].strip()
    print(spit_out)
    return spit_out

prompt_list = ['Revise this with your best effort', 'Help me polish this', 'Rewrite this for me', 
                'Make this fluent while doing minimal change', 'Refine this for me please', 'Concise this for me and keep all the information',
                'Improve this in GPT way']



# with open(f'yelp_GPT-evade_Help_me_rephrase_it_in_human_style_-400sub.json', 'r') as file:
#     GPT = json.load(file)

with open(f'yelp_GPT-evade_Help_me_rephrase_it,_so_that_another_GPT_rewriting_will_cause_a_lot_of_modifications-400sub.json', 'r') as file:
    GPT = json.load(file)

def rewrite_json(input_json, prompt_list, human=False):
    all_data = []
    for cc, data in enumerate(input_json):
        tmp_dict ={}
        
        tmp_dict['input'] = data

        for ep in prompt_list:
            tmp_dict[ep] = GPT_self_prompt(ep, tmp_dict['input'])
        
        all_data.append(tmp_dict)

        if debug:
            break
    return all_data


GPT_rewrite = rewrite_json(GPT, prompt_list)
# with open(f'rewrite_arxiv_GPT_inv_yelp_GPT-evade_Help_me_rephrase_it_in_human_style_-400sub.json', 'w') as file:
#     json.dump(GPT_rewrite, file, indent=4)

with open(f'rewrite_arxiv_GPT_inv_yelp_GPT-evade_Help_me_rephrase_it,_so_that_another_GPT_rewriting_will_cause_a_lot_of_modifications-400sub.json', 'w') as file:
    json.dump(GPT_rewrite, file, indent=4)
