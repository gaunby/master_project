import numpy as np
from datasets import load_from_disk
import pandas as pd
import pickle
import sys
sys.path.insert(0,'/zhome/94/5/127021/speciale/master_project')
from src.models.hate_tcav.TCAV import get_preds_tcavs

# load
N = 300

filename = "/work3/s174498/sst2_dataset/positive"
ds_pos = load_from_disk(filename)
ds_pos_text = ds_pos['sentence']

filename = "/work3/s174498/sst2_dataset/negative"
ds_neg = load_from_disk(filename)
ds_neg_text = ds_neg['sentence']

pos = [ds_pos_text[i] for i in list(np.random.choice(len(ds_pos_text),N))]
neg = [ds_neg_text[i] for i in list(np.random.choice(len(ds_neg_text),N))]

# Concept data 
M = 150
# load hate
datadir = '/work3/s174498/concept_random_dataset/'
filename = 'tweet_hate/test'
ds_hate = load_from_disk(datadir + filename)
ds_hate = ds_hate['text']
hate = [ds_hate[i] for i in list(np.random.choice(len(ds_hate),M))]

# load 20 newsgroups
filename = '20_newsgroups/test'
ds_news= load_from_disk(datadir + filename)
ds_news = ds_news['text']
news = [ds_news[i] for i in list(np.random.choice(len(ds_news),M))]

# load ag news 
# labels: World (0), Sports (1), Business (2), Sci/Tech (3).
filename = 'ag_news/test'
ag_news= load_from_disk(datadir + filename)

df_label_ag = pd.DataFrame(ag_news['label'])
idx_world = df_label_ag[df_label_ag[0] == 0].index.values
idx_sport = df_label_ag[df_label_ag[0] == 1].index.values
idx_buss = df_label_ag[df_label_ag[0] == 2].index.values
idx_sci = df_label_ag[df_label_ag[0] == 3].index.values

ag_news = ag_news['text']
ag_world = [ag_news[i] for i in list(np.random.choice( idx_world,M))]
ag_sport = [ag_news[i] for i in list(np.random.choice( idx_sport,M))]
ag_buss = [ag_news[i] for i in list(np.random.choice( idx_buss,M))]
ag_sci = [ag_news[i] for i in list(np.random.choice( idx_sci,M))]


num_random_set = 500
model_layer = 'roberta.encoder.layer.11.output.dense'
layer_nr = '11'
logits,sensitivity,TCAV, acc = get_preds_tcavs(classifier = 'linear',model_layer=model_layer,layer_nr =layer_nr,
                                    target_text = neg, desired_class=0,
                                    counter_set = 'wikipedia_split',
                                    concept_text = hate, 
                                    num_runs=num_random_set)
save_tcav = {}
save_tcav['negative'] = {'hate':0}
save_tcav['negative']['hate'] = {model_layer:0}
save_tcav['negative']['hate'][model_layer] = {'TCAV': TCAV, 'acc': acc}
print(save_tcav)

concept = 'news'
logits,sensitivity,TCAV, acc = get_preds_tcavs(classifier = 'linear',model_layer=model_layer,layer_nr =layer_nr,
                                    target_text = neg, desired_class=0,
                                    counter_set = 'wikipedia_split',
                                    concept_text = news, 
                                    num_runs=num_random_set)
save_tcav['negative'][concept] = {model_layer:0}
save_tcav['negative'][concept][model_layer] = {'TCAV': TCAV, 'acc': acc}

concept = 'sports'
logits,sensitivity,TCAV, acc = get_preds_tcavs(classifier = 'linear',model_layer=model_layer,layer_nr =layer_nr,
                                    target_text = neg, desired_class=0,
                                    counter_set = 'wikipedia_split',
                                    concept_text = ag_sport, 
                                    num_runs=num_random_set)
save_tcav['negative'][concept] = {model_layer:0}
save_tcav['negative'][concept][model_layer] = {'TCAV': TCAV, 'acc': acc}

concept = 'sci'
logits,sensitivity,TCAV, acc = get_preds_tcavs(classifier = 'linear',model_layer=model_layer,layer_nr =layer_nr,
                                    target_text = neg, desired_class=0,
                                    counter_set = 'wikipedia_split',
                                    concept_text = ag_sci, 
                                    num_runs=num_random_set)
save_tcav['negative'][concept] = {model_layer:0}
save_tcav['negative'][concept][model_layer] = {'TCAV': TCAV, 'acc': acc}

file_name = 'negative_layer_11'
PATH =  f"/work3/s174498/nlp_tcav_results/{file_name}.pkl"
f = open(PATH ,"wb")
pickle.dump(save_tcav, f)
f.close()