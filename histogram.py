
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from fuzzywuzzy import fuzz


from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

def tokenize_and_normalize(sentence):
    # Tokenization and normalization
    return [word.lower().strip() for word in sentence.split()]

def extract_ngrams(tokens, n):
    # Extract n-grams from the list of tokens
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def common_elements(list1, list2):
    # Find common elements between two lists
    return set(list1) & set(list2)
def calculate_sentence_common(sentence1, sentence2):
    tokens1 = tokenize_and_normalize(sentence1)
    tokens2 = tokenize_and_normalize(sentence2)

    # Find common words
    common_words = common_elements(tokens1, tokens2)

    # Find common n-grams (let's say up to 3-grams for this example)
    common_ngrams = set()
    

    number_common_hierarchy = [len(list(common_words))]

    for n in range(2, 5):  # 2-grams to 3-grams
        ngrams1 = extract_ngrams(tokens1, n)
        ngrams2 = extract_ngrams(tokens2, n)
        common_ngrams = common_elements(ngrams1, ngrams2) 
        number_common_hierarchy.append(len(list(common_ngrams)))

    return number_common_hierarchy


data = 'yelp'
with open('Yelp/rewrite_yelp_GPT_inv.json', 'r') as f:
    data_gpt = json.load(f)
with open('Yelp/rewrite_yelp_human_inv.json', 'r') as f:
    data_human = json.load(f)


prompt_list = ['Revise this with your best effort', 'Help me polish this', 'Rewrite this for me', 
                'Make this fluent while doing minimal change', 'Refine this for me please', 'Concise this for me and keep all the information',
                'Improve this in GPT way']

# prompt_list = ['Revise the code with your best effort', 'Help me polish this code', 'Rewrite the code with GPT style', 
# 'Refine the code for me please', 'Concise the code without change the functionality'] # invariance


for prompt_name in prompt_list:

    ngram_num = 4
    def sum_for_list(a,b):
        return [aa+bb for aa, bb in zip(a,b)]

    cutoff_start = 0
    cutoff_end = 6000000

    def get_data_stat(data_json):
        total_len = len(data_json)

        for idxx, each in enumerate(data_json):
            original = each['input']
            raw = tokenize_and_normalize(each['input'])
            if len(raw)<cutoff_start or len(raw)>cutoff_end:
                continue
            else:
                print(idxx, total_len)

            statistic_res = {}
            ratio_fzwz = {}
            all_statistic_res = [0 for i in range(ngram_num)]
            cnt = 0
            whole_combined=''
            for pp in each.keys():
                if pp != 'common_features' and pp == prompt_name:
                    whole_combined += (' ' + each[pp])
                    

                    res = calculate_sentence_common(original, each[pp])
                    statistic_res[pp] = res
                    all_statistic_res = sum_for_list(all_statistic_res, res)

                    ratio_fzwz[pp] = [fuzz.ratio(original, each[pp]), fuzz.token_set_ratio(original, each[pp])]
                    # import pdb; pdb.set_trace()
                    cnt += 1
            
            each['fzwz_features'] = ratio_fzwz
            each['common_features'] = statistic_res
            # each['avg_common_features'] = [a/cnt for a in all_statistic_res]

            each['common_features_ori_vs_allcombined'] = calculate_sentence_common(original, whole_combined)

        return data_json
  
    gpt = get_data_stat(data_gpt)
    human = get_data_stat(data_human)

    def plot_hist(gpt, human):
        

        def get_value(input_json, prompt_name):
            ans = []
            for each in input_json:
                # import pdb; pdb.set_trace()
                # try:
                ans.append(each['fzwz_features'][prompt_name][0])
                # ans.append(each['common_features'][prompt_name][1])
                    # print(each['fzwz_features'][prompt_name][0])
                # except:
                #     break
                    # import pdb; pdb.set_trace()
            return ans
        
        h_list = get_value(human, prompt_name)
        g_list = get_value(gpt, prompt_name)

        # import pdb; pdb.set_trace()

        def plot_histogram(list1, list2, roc_auc):
            # Set the transparency level (alpha)
            # import pdb; pdb.set_trace()
            # print(len(list1), len(list2))
            alpha_value = 0.5
            num_bins = 100
            # num_bins=20

            # Plot the histograms
            plt.hist(list1, bins=num_bins,alpha=alpha_value, color='blue', label='human')
            plt.hist(list2, bins=num_bins,alpha=alpha_value, color='red', label='GPT')

            # Set the labels and title
            plt.xlabel('Similarity')
            plt.ylabel('Count')
            plt.title('Histogram of Rewriting Consistency')
            plt.legend()

            # Add grid
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)


            # Display the plot
            plt.show()
            os.makedirs(f'eps2/{data}', exist_ok = True)
            plt.savefig(f'eps2/{data}/' + prompt_name+f'_{data}_thre_detect_{roc_auc:.2f}.eps', format='eps')
            plt.clf()
        
        

        a = np.array(h_list)
        b = np.array(g_list)
        labels = np.concatenate((np.zeros_like(a), np.ones_like(b)), axis=0)
        prob = np.concatenate((a,b), axis=0)
        fpr, tpr, thresholds = roc_curve(labels, prob)
        roc_auc = auc(fpr, tpr)
        t= f'ROC curve (area = {roc_auc:.2f})'
        # axs[4].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        print(roc_auc, data, prompt_name)

        plot_histogram(h_list, g_list, roc_auc)


    min_len = min(len(gpt), len(human))
    human = human[:min_len]

    # print(gpt)
    # import pdb; pdb.set_trace()
    plot_hist(gpt, human)



            


