

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


with open('../rewrite_yelp_GPT_inv.json', 'r') as f:  # XGBoost best
    data_gpt = json.load(f)
with open('rewrite_arxiv_GPT_inv_yelp_GPT-evade_Help_me_rephrase_it_in_human_style_-400sub.json', 'r') as f:  # XGBoost best
    data_gpt_bypass = json.load(f)
with open('rewrite_arxiv_GPT_inv_yelp_GPT-evade_Help_me_rephrase_it,_so_that_another_GPT_rewriting_will_cause_a_lot_of_modifications-400sub.json', 'r') as f:  # XGBoost best
    data_gpt_bypass2 = json.load(f)
with open('../rewrite_yelp_human_inv.json', 'r') as f:
    data_human = json.load(f)


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

    
gpt = get_data_stat(data_gpt)
gpt_bypass = get_data_stat(data_gpt_bypass)
gpt_bypass2 = get_data_stat(data_gpt_bypass2)
human = get_data_stat(data_human)

def xgboost_classifier(gpt, human, gpt_bypass, gpt_bypass2):

    def get_feature_vec(input_json):
        all_list = []
        for each in input_json:
            
            try:
                raw = tokenize_and_normalize(each['input'])
                r_len = len(raw)*1.0
            except:
                import pdb; pdb.set_trace()
            if r_len ==0:
                continue

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

        return all_list
    
    gpt_all = get_feature_vec(gpt)
    human_all = get_feature_vec(human)
    gpt_bypass = get_feature_vec(gpt_bypass)
    gpt_bypass2 = get_feature_vec(gpt_bypass2)

    # import pdb; pdb.set_trace() # dim 112,28 

    # # random split, may have content similarity   
    # gpt_all = np.concatenate((gpt_all, gpt_all), axis=0) 

    # ### Original
    # X = np.concatenate((gpt_all, human_all), axis=0)
    # Y = np.concatenate((np.ones(gpt_all.shape[0]), np.zeros(human_all.shape[0])), axis=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


    # reblanced
    g_train, g_test, yg_train, yg_test = train_test_split(gpt_all, np.ones(gpt_all.shape[0]), test_size=0.2, random_state=42)
    h_train, h_test, yh_train, yh_test = train_test_split(human_all, np.zeros(human_all.shape[0]), test_size=0.2, random_state=42)

    gbp_train, gbp_test, ygbp_train, ygbp_test = train_test_split(gpt_bypass, np.ones(gpt_bypass.shape[0]), test_size=0.2, random_state=42)
    gbp_train2, gbp_test2, ygbp_train2, ygbp_test2 = train_test_split(gpt_bypass2, np.ones(gpt_bypass2.shape[0]), test_size=0.2, random_state=42)

    X_train = np.concatenate((g_train, gbp_train2, h_train), axis=0)
    y_train = np.concatenate((yg_train, ygbp_train2, yh_train), axis=0)

    # X_train = np.concatenate((g_train, h_train), axis=0)
    # y_train = np.concatenate((yg_train, yh_train), axis=0)

    # X_train = np.concatenate((g_train, g_train, g_train, h_train), axis=0)
    # y_train = np.concatenate((yg_train, yg_train, yg_train, yh_train), axis=0)

    # X_test = np.concatenate((g_test, h_test), axis=0) # In distribution test
    # y_test = np.concatenate((yg_test, yh_test), axis=0)

    # X_test = np.concatenate((h_test, gbp_test), axis=0)  # Out of distribution, active bypass test
    # y_test = np.concatenate((yh_test, ygbp_test), axis=0)


    # X_train = np.concatenate((gbp_train, gbp_train2, h_train), axis=0)
    # y_train = np.concatenate((ygbp_train, ygbp_train2, yh_train), axis=0)

    X_train = np.concatenate((g_train, h_train), axis=0)
    y_train = np.concatenate((yg_train, yh_train), axis=0)

    # ######
    # X_train = np.concatenate((gbp_train, gbp_train2, h_train), axis=0)
    # y_train = np.concatenate((ygbp_train, ygbp_train2, yh_train), axis=0)
    # X_test = np.concatenate((g_test, h_test), axis=0) # In distribution test
    # y_test = np.concatenate((yg_test, yh_test), axis=0)

    # ######


    # X_train = np.concatenate((g_train, gbp_train, h_train), axis=0)
    # y_train = np.concatenate((yg_train, ygbp_train, yh_train), axis=0)
    # # # X_train = np.concatenate((g_train, h_train), axis=0)
    # # # y_train = np.concatenate((yg_train, yh_train), axis=0)
    X_test = np.concatenate((h_test, gbp_test2), axis=0)  # Out of distribution, active bypass test
    y_test = np.concatenate((yh_test, ygbp_test2), axis=0)


    X_test = np.concatenate((h_test, gbp_test), axis=0)  # Out of distribution, active bypass test
    y_test = np.concatenate((yh_test, ygbp_test), axis=0)


    # X_train = np.concatenate((g_train, gbp_train2, h_train), axis=0)
    # y_train = np.concatenate((yg_train, ygbp_train2, yh_train), axis=0)

    # X_test = np.concatenate((h_test, gbp_test), axis=0)  # Out of distribution, active bypass test
    # y_test = np.concatenate((yh_test, ygbp_test), axis=0)



    # # block split,  leads to 72% acc
    # gpt_all_train_len = int(gpt_all.shape[0] * 0.8)
    # human_all_train_len = int(human_all.shape[0] * 0.8)
    # gpt_train = gpt_all[:gpt_all_train_len]
    # gpt_test = gpt_all[gpt_all_train_len:]
    # human_train = human_all[:human_all_train_len]
    # human_test = human_all[human_all_train_len:]
    # X_train = np.concatenate((gpt_train, human_train), axis=0)
    # y_train = np.concatenate((np.ones(gpt_train.shape[0]), np.zeros(human_train.shape[0])), axis=0)

    # X_test = np.concatenate((gpt_test, human_test), axis=0)
    # y_test = np.concatenate((np.ones(gpt_test.shape[0]), np.zeros(human_test.shape[0])), axis=0)

    #### 

    # # Create and train the model
    model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=10, random_state=42) # 74.44\%,  turn out for reuter: 62%
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # # # # Logistic regression
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test) # 66.22

    # print(y_pred)

    # # Neural network
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # # # clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam', random_state=42) # 75.58
    # # clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='relu', solver='adam', random_state=42) # 75.83, using fuzzywazzy, get 78.5% acc.
    # # # clf = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000, activation='relu', solver='adam', random_state=42) # 71.41
    # # clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)  # 74.38
    # # # clf = RandomForestClassifier(n_estimators=100, random_state=42) # 73.37
    # clf = KNeighborsClassifier(n_neighbors=3) # 67.93
    # # # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=42) # 72.04
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)

    from sklearn.metrics import accuracy_score, classification_report, f1_score

    print("Accuracy:", accuracy_score(y_test, y_pred), "F1 score", f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

xgboost_classifier(gpt, human, gpt_bypass, gpt_bypass2)


    

        
        
        









            


