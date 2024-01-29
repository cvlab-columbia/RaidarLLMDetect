
# Only ::5 is needed.

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


evade_str_list = ["Help me rephrase it in human style ",
                    "Help me rephrase it, so that another GPT rewriting will cause a lot of modifications"]

debug=False
with open(f'../arxiv_GPT_concise.json', 'r') as file:
    GPT = json.load(file)

for evade_str in evade_str_list:
    def GPT_self_prompt(code):

        response = openai_backoff(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "user",
                                "content": f" : {code}",
                            }
                        ],
                    )
        spit_out = response["choices"][0]["message"]["content"].strip()
        print(spit_out)
        return spit_out

    

    code = []
    for each in GPT:
        output = GPT_self_prompt(each['abs'])
        code.append({"abs":  output, "title": each['title']})
        if debug:
            break

    evade_str = evade_str.replace(' ', '_')
    with open(f'arXiv_GPT-evade_{evade_str}.json', 'w') as file:
        json.dump(code, file, indent=4)





