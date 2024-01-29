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

with open(f'arXiv_human.json', 'r') as file:
    arxiv = json.load(file)

def tokenize_and_normalize(sentence):
    # Tokenization and normalization
    return [word.strip() for word in sentence.split()]


def GPT_self_prompt(title, abst):

    abst_list = tokenize_and_normalize(abst)[:15]
    prompt_str = ''.join(abst_list)
    print('*******', prompt_str)

    response = openai_backoff(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": f"The title is {title}, start with {prompt_str}, write a short concise abstract based on this: ", # yelp GPT concise
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
for each in arxiv:
    ans={}
    output = GPT_self_prompt(each['title'], each['abs'])
    ans['abs'] = output
    ans['title'] = each['title']
    GPT.append(ans)
    # break

with open(f'arxiv_GPT_concise.json', 'w') as file:
    json.dump(GPT, file, indent=4)



