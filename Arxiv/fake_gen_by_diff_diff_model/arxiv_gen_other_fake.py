import os
import openai
import json

openai.api_key='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'

# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_random_exponential,
# )

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
# def openai_backoff(**kwargs):
#     return openai.Completion.create(**kwargs)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

with open(f'../arXiv_human.json', 'r') as file:
    arxiv = json.load(file)

def tokenize_and_normalize(sentence):
    # Tokenization and normalization
    return [word.strip() for word in sentence.split()]


modeltype_list = ["text-davinci-002", 'ada']
modeltype_list = ['gpt-4-1106-preview']

debug=False
max_token = 300

for modeltype in modeltype_list:

    def GPT_self_prompt(title, abst):

        abst_list = tokenize_and_normalize(abst)[:15]
        prompt_str = ''.join(abst_list)
        print('*******', prompt_str)

        if 'gpt-4' in modeltype:
            response = openai_backoff(
                        model=modeltype,
                        messages=[
                            {
                                "role": "user",
                                "content": f"The title is {title}, start with {prompt_str}, write a short concise abstract based on this: ", # yelp GPT concise
                            }
                        ],
                    )
            spit_out = response["choices"][0]["message"]["content"].strip()
        else:
            response = openai_backoff(
                            model=modeltype,
                            prompt=f"The title is {title}, start with {prompt_str}, write a short concise abstract based on this: ",
                            temperature=0,
                            max_tokens =max_token,
                        )
            # spit_out = response["choices"][0]["message"]["content"].strip()
            # import pdb; pdb.set_trace()
            spit_out = response["choices"][0]["text"]

        print(spit_out)
        return spit_out


    GPT = []
    for each in arxiv:
        ans={}
        output = GPT_self_prompt(each['title'], each['abs'])
        ans['abs'] = output
        ans['title'] = each['title']
        GPT.append(ans)
        if debug:
            break

    with open(f'{modeltype}_arxiv_GPT_concise.json', 'w') as file:
        json.dump(GPT, file, indent=4)



