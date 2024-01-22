# only use the first 400 is fine

# since 2000 in total, 20% for test

import os
import openai
import json

openai.api_key='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'

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# for gpt ada, 3.
# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
# def openai_backoff(**kwargs):
#     return openai.Completion.create(**kwargs)

# for 3.5 and above
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

with open(f'../yelp_subset.json', 'r') as file:
    yelp = json.load(file)


modeltype_list = ["text-davinci-002", 'ada'] 
modeltype_list = ['gpt-4-1106-preview']

debug=False
max_token = 200
save_interval=20

for modeltype in modeltype_list:
    def GPT_self_prompt(prompt_str):
        if 'gpt-4' in modeltype:
            response = openai_backoff(
                    model=modeltype,
                    messages=[
                        {
                            "role": "user",
                            # "content": f"Help me write a review based on this with similar length: {prompt_str}",
                            "content": f"Write a very short and concise review based on this: {prompt_str}",
                        }
                    ],
                )
            spit_out = response["choices"][0]["message"]["content"].strip()
        else:
            response = openai_backoff(
                model=modeltype,
                            prompt=f"Write a Yelp review based on this: {prompt_str}",
                            temperature=0,
                            max_tokens =max_token,
            )

            spit_out = response["choices"][0]["text"].strip()
            print(spit_out)
            if spit_out == '':
                response = openai_backoff(
                model=modeltype,
                            prompt=f"Write an expanded Yelp review {prompt_str}, just write something:",
                            temperature=0,
                            max_tokens =max_token,
                )

                spit_out = response["choices"][0]["text"].strip()
            
        return spit_out


    GPT = []
    cnt = 0
    cutoff=400
    for each in yelp:
        output = GPT_self_prompt(each['text'])
        # import pdb; pdb.set_trace()
        GPT.append(output)
        cnt += 1
        if cnt == cutoff:
            break

        if debug:
            break

        if cnt % save_interval==0:
            with open(f'{modeltype}_yelp_GPT_concise_400test_only.json', 'w') as file:
                json.dump(GPT, file, indent=4)

    # print('GPT', GPT)
    with open(f'{modeltype}_yelp_GPT_concise_400test_only.json', 'w') as file:
        json.dump(GPT, file, indent=4)










