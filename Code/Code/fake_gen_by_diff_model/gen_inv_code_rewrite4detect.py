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

prompt_list = ['Revise the code with your best effort', 'Help me polish this code', 'Rewrite the code with GPT style', 'Refine the code for me please', 'Concise the code without change the functionality'] # invariance
prefix='. No need to explain. Just write code:'

filelist = [('ada_code_GPT-v2.json', 'rewrite_code_GPT_inv-ada_.json'),
('text-davinci-002_code_GPT-v2.json', 'rewrite_code_GPT_inv-text-davinci-002.json')]

filelist = [('gpt-4-1106-preview_code_GPT-v2.json', 'rewrite_code_GPT_inv-GPT4_.json')]

filelist = [('llama2_7b_chat_code_GPT-v2.json', 'rewrite_code_inv-llama2-7bchat.json')]

for inputname, outputname in filelist:
    with open(inputname, 'r') as file:
        GPT = json.load(file)


    def rewrite_json(input_json, prompt_list, human=False):
        all_data = []
        for cc, data in enumerate(input_json):
            tmp_dict ={}
            if human:
                tmp_dict['input'] = data[0] + data[1] # prmpot + solution, should be one, but saved separately.
            else:
                tmp_dict['input'] = data[1]  # this is the GPT rewritten, which already contain prompt and solution

            for ep in prompt_list:
                tmp_dict[ep] = GPT_self_prompt(ep, tmp_dict['input'], prefix)
            
            all_data.append(tmp_dict)

            if debug:
                break
        return all_data



    GPT_rewrite = rewrite_json(GPT, prompt_list)
    with open(outputname, 'w') as file:
        json.dump(GPT_rewrite, file, indent=4)


