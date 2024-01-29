import os
import openai
import json

openai.api_key='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'

# for Completaion
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_random_exponential,
# )

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
# def openai_backoff(**kwargs):
#     return openai.Completion.create(**kwargs)

# for chat model, gpt-3.5, 4
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)




codedata=[]
with open(f'../HumanEval.jsonl', 'r') as file:
    for line in file:
        codedata.append(json.loads(line))
        # break

modeltype_list = ["text-davinci-002", 'ada']
modeltype_list = ['gpt-4-1106-preview']

debug=False
max_token = 300

for modeltype in modeltype_list:
    def GPT_self_prompt(prompt_str, solution):
        if 'gpt-4' not in modeltype:
            response = openai_backoff(
                            model=modeltype,
                            prompt= f"Describe what does this code do: {prompt_str}{solution}",
                            temperature=0,
                            max_tokens =max_token,
                        )
            ans = response["choices"][0]["text"].strip()
            # print('pseudo code', ans)

            response = openai_backoff(
                            model=modeltype,
                            prompt= f"I want to do this: {ans} Help me write python code start with this {prompt_str}, no explanation, just code:",
                            temperature=0,
                            max_tokens =max_token,
                        )
            spit_out = response["choices"][0]["text"].strip()
        else:
            response = openai_backoff(
                            model=modeltype,
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"Describe what does this code do: {prompt_str}{solution}",
                                }
                            ],
                        )
            ans = response["choices"][0]["message"]["content"].strip()
            # print('pseudo code', ans)

            response = openai_backoff(
                            model=modeltype,
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"I want to do this: {ans} Help me write python code start with this {prompt_str}, no explanation, just code:",
                                }
                            ],
                        )
            spit_out = response["choices"][0]["message"]["content"].strip()


        print(spit_out)
        return spit_out, ans

    code = []
    for each in codedata:
        output, ans = GPT_self_prompt(each['prompt'], each['canonical_solution'])
        code.append((each['prompt'], output, ans))
        if debug:
            break

    with open(f'{modeltype}_code_GPT-v2.json', 'w') as file:
        json.dump(code, file, indent=4)
