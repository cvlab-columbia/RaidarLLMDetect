import os
import openai
import json

with open(f'code_human-v2.json', 'r') as file:
    human_raw = json.load(file)

with open(f'rewrite_code_human_inv.json', 'r') as file:
    human = json.load(file)

for i in range(len(human)):
    human[i]['input'] = human_raw[i][0] + human_raw[i][1] 


with open(f'rewrite_code_human_inv2.json', 'w') as file:
    json.dump(human, file, indent=4)


