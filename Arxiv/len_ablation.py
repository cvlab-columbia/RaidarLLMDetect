
import json
import matplotlib.pyplot as plt
import numpy as np

from fuzzywuzzy import fuzz


from sklearn.metrics import roc_curve, auc
import xgboost as xgb
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

sentence1 = "I love to play football in the park."
sentence2 = "He loves to play in the park with friends."


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


with open('rewrite_arxiv_GPT_inv.json', 'r') as f: # get 77% for those features solely.
    data_gpt = json.load(f)
with open('rewrite_arxiv_human_inv.json', 'r') as f:
    data_human = json.load(f)

# 

# # News
# with open('reuter_gpt_para_alllen_ours_0-1221.json', 'r') as f:  # XGBoost best
#     data_gpt = json.load(f)
# with open('reuter_human_para_alllen_ours_0-1221.json', 'r') as f:
#     data_human = json.load(f)

# # essay
# with open('iclr_essay_gpt_para_comb_300.json', 'r') as f:  # XGBoost best
#     data_gpt = json.load(f)
# with open('iclr_essay_human_para_comb_300.json', 'r') as f:
#     data_human = json.load(f)

# with open('v2-essay_gpt_para_ours_0-300.json', 'r') as f: # get 77% for those features solely.
#     data_gpt = json.load(f)
# with open('v2-essay_human_para_ours_0-300.json', 'r') as f:
#     data_human = json.load(f)

# with open('essay_gpt_para_ours_Equi_0-300.json', 'r') as f: # equi, get 78.621%
#     data_gpt = json.load(f)
# with open('essay_human_para_ours_Equi_0-300.json', 'r') as f:
#     data_human = json.load(f)


ngram_num = 4
def sum_for_list(a,b):
    return [aa+bb for aa, bb in zip(a,b)]

cutoff_start = 0
cutoff_end = 6000000

def get_data_stat(data_json):
    total_len = len(data_json)
    for idxx, each in enumerate(data_json):
        
        original = each['input']

        # remove too short ones
        
        # import pdb; pdb.set_trace()
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
            if pp != 'common_features':
                whole_combined += (' ' + each[pp])
                

                res = calculate_sentence_common(original, each[pp])
                statistic_res[pp] = res
                all_statistic_res = sum_for_list(all_statistic_res, res)

                ratio_fzwz[pp] = [fuzz.ratio(original, each[pp]), fuzz.token_set_ratio(original, each[pp])]
                cnt += 1
        
        each['fzwz_features'] = ratio_fzwz
        each['common_features'] = statistic_res
        each['avg_common_features'] = [a/cnt for a in all_statistic_res]

        each['common_features_ori_vs_allcombined'] = calculate_sentence_common(original, whole_combined)

    return data_json

def get_data_len_diff(data_json):

    diff = []
    for each in data_json:
        original = len(tokenize_and_normalize(each['input']))

        cnt = 0
        t_len = 0
        for pp in each.keys():
            if pp != 'common_features' and pp!='avg_common_features' and pp!='common_features_ori_vs_allcombined' and pp!='fzwz_features':
                # import pdb; pdb.set_trace()
                try:
                    t_len += len(tokenize_and_normalize(each[pp]))
                    cnt += 1
                except:
                    import pdb; pdb.set_trace()
        
        diff.append(t_len/cnt - original)
        each['diff'] = t_len/cnt - original

    return data_json


    
gpt = get_data_stat(data_gpt)
human = get_data_stat(data_human)
# import pdb; pdb.set_trace()

gpt = get_data_len_diff(gpt)
human = get_data_len_diff(human)


def xgboost_classifier(gpt, human):

    def get_feature_vec(input_json):
        len_list = []
        all_list = []
        for each in input_json:
            
            try:
                raw = tokenize_and_normalize(each['input'])
                len_list.append(len(raw))
                r_len = len(raw)*1.0
            except:
                import pdb; pdb.set_trace()
            each_data_fea  = []
            if len(raw)<cutoff_start or len(raw)>cutoff_end:
                continue

            # each_data_fea  = [len(raw) / 100.]
            
            each_data_fea = [ind_d / r_len for ind_d in each['avg_common_features']]
            for ek in each['common_features'].keys():
                each_data_fea.extend([ind_d / r_len for ind_d in each['common_features'][ek]])
            
            each_data_fea.extend([ind_d / r_len for ind_d in each['common_features_ori_vs_allcombined']])

            for ek in each['fzwz_features'].keys():
                each_data_fea.extend(each['fzwz_features'][ek])

            all_list.append(np.array(each_data_fea))
        all_list = np.vstack(all_list)

        len_list = np.array(len_list)

        return all_list, len_list
    
    gpt_all, g_len = get_feature_vec(gpt)
    human_all, h_len = get_feature_vec(human)
    # import pdb; pdb.set_trace() # dim 112,28 

    # # random split, may have content similarity   
    # gpt_all = np.concatenate((gpt_all, gpt_all), axis=0) 

    # ### Original
    # X = np.concatenate((gpt_all, human_all), axis=0)
    # Y = np.concatenate((np.ones(gpt_all.shape[0]), np.zeros(human_all.shape[0])), axis=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


    # reblanced
    g_train, g_test, yg_train, yg_test, gl_train, gl_test = train_test_split(gpt_all, np.ones(gpt_all.shape[0]), g_len, test_size=0.2, random_state=42)
    h_train, h_test, yh_train, yh_test, hl_train, hl_test = train_test_split(human_all, np.zeros(human_all.shape[0]), h_len, test_size=0.2, random_state=42)

    X_train = np.concatenate((g_train, g_train, h_train), axis=0)
    y_train = np.concatenate((yg_train, yg_train, yh_train), axis=0)


    X_test = np.concatenate((g_test, h_test), axis=0)
    y_test = np.concatenate((yg_test, yh_test), axis=0)

    len_test = np.concatenate((gl_test, hl_test), axis=0)

    # vizlen = len_test.tolist()
    # plt.hist(vizlen, bins=10, edgecolor='black', alpha=0.7)

    # # Add title and labels
    # plt.title('Histogram of Data')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.savefig('zzz.jpeg')

    # exit()


    # # # Logistic regression
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test) # 66.22

    # # Neural network
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # # # clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam', random_state=42) # 75.58
    # # clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='relu', solver='adam', random_state=42) # 75.83, using fuzzywazzy, get 78.5% acc.
    # # # clf = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000, activation='relu', solver='adam', random_state=42) # 71.41
    # # clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)  # 74.38
    # # # clf = RandomForestClassifier(n_estimators=100, random_state=42) # 73.37
    # # # clf = KNeighborsClassifier(n_neighbors=3) # 67.93
    # # # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=42) # 72.04
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    # import pdb; pdb.set_trace()
    print(min(len_test)) # 
    print(max(len_test)) # 

    ans = []
    x = []
    amount = []
    for start, finish in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120),(120, 150), (150,180), (180,200)]: 
        sub_y_test = []
        sub_y_pred = []

        for jj in range(len(len_test)):
            l = len_test[jj]
            if l>start and l<=finish:
                sub_y_test.append(y_test[jj])
                sub_y_pred.append(y_pred[jj])

        print(start, finish)
        print("Accuracy:", accuracy_score(sub_y_test, sub_y_pred), "F1 score", f1_score(sub_y_test, sub_y_pred))
        print(classification_report(sub_y_test, sub_y_pred))
        ans.append(f1_score(sub_y_test, sub_y_pred))
        x.append((start+finish)/2)
        amount.append(len(sub_y_pred))
    

    plt.figure()
    plt.plot(x, ans, '-o', color='blue', label='Detection via Rewriting')

    total_amt = sum(amount)
    percentage_amount = [int(each*1.0/total_amt * 2000) for each in amount]
    plt.scatter(x, ans, s=percentage_amount, alpha=0.5)  # Add scatter plot
    for i in range(len(x)):
        plt.text(x[i], ans[i], str(amount[i]), fontsize=9, ha='right', va='bottom')

    plt.title('arXiv GPT Detection')
    plt.xlabel('Text Length')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('arxiv_len_ablation.eps')

    print('am', amount) #





xgboost_classifier(gpt, human)


    

        
        
        









            
