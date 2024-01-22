import os
import json

codedata=[]
cnt=0
with open('yelp_academic_dataset_review.json', 'r') as file:
    for line in file:
        codedata.append(json.loads(line))
        cnt += 1
        if cnt == 2000:
            break

with open('yelp_subset.json', 'w') as file:
    json.dump(codedata, file, indent=4)

