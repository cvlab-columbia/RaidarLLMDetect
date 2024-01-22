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

with open(f'yelp_subset.json', 'r') as file:
    yelp = json.load(file)

def GPT_self_prompt(prompt_str):


    response = openai_backoff(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            # "content": f"Help me write a review based on this with similar length: {prompt_str}",
                            "content": f"Write a very short and concise review based on this: {prompt_str}",
                        }
                    ],
                )
    spit_out = response["choices"][0]["message"]["content"].strip()
    print(spit_out)
    return spit_out

# human = []
# for each in yelp:
#     human.append(each['text'])

# with open(f'yelp_human.json', 'w') as file:
#     json.dump(human, file, indent=4)

GPT = []
for each in yelp:
    output = GPT_self_prompt(each['text'])
    GPT.append(output)
    # break

with open(f'yelp_GPT_concise.json', 'w') as file:
    json.dump(GPT, file, indent=4)



