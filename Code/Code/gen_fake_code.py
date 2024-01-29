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


codedata=[]
with open(f'HumanEval.jsonl', 'r') as file:
    for line in file:
        codedata.append(json.loads(line))
        # break

# for each in codedata:
#     # print(each.keys())
#     # print(each['prompt'])
#     # print('#######')
#     # print(each['entry_point'])
#     # print('#######')
#     # print(each['canonical_solution'])
#     # break



# Can also try GPT 4 generated code. 
# Then claim we can also detect GPT 4. Can only do test. No need to train. 

def GPT_self_prompt(prompt_str, solution):

    response = openai_backoff(
                    model="gpt-3.5-turbo",
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
                    model="gpt-3.5-turbo",
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

# prompt_list = ['Revise this with your best effort', 'Help me polish this', 'Rewrite this for me', 'Make this fluent while doing minimal change'] # invariance
# prompt_list = ['Refine this for me please', 'Concise this for me and keep all the information']
# cycle consistency, equivariant
# brainstorm: write this in the opposite meaning? Then write the opposite again in opposite? (cycle consistency)  refine this? concise this?
# expand this by 1 time. Concise this by 50%
# translate this to chineses, then translate this back to Enligh.

# TODO: ask GPT to tell, if those two sentence are written by the same guy or no. As a feature
# TODO: equivariant: Expand this by 50 words. Concise this by 50 words.
# TODO: 

human = []
for each in codedata:
    human.append((each['prompt'], each['canonical_solution']))

with open(f'code_human-v2.json', 'w') as file:
    json.dump(human, file, indent=4)

code = []
for each in codedata:
    output, ans = GPT_self_prompt(each['prompt'], each['canonical_solution'])
    code.append((each['prompt'], output, ans))

with open(f'code_GPT-v2.json', 'w') as file:
    json.dump(code, file, indent=4)
