import numpy as np
from datasets import load_from_disk
import pandas as pd
import pickle
import sys
sys.path.insert(0,'/zhome/94/5/127021/speciale/master_project')
from src.models.hate_tcav.TCAV import get_preds_tcavs


# PARAMETERS
FILE_NAME = 'negative_gender_layer_0_9' # name of saved file 
N = 300 # number of target examples 
M = 150 # number of concept examples

num_random_set = 100 # number of runs/random folders

concepts = ['hate'] # if not hate or news set variable later on 
concepts = ['Woman','Transsexual','Intersex']
target_nr = 0 
target_name = 'negative'


# target data 
# load
filename = "/work3/s174498/sst2_dataset/positive"
ds_pos = load_from_disk(filename)
ds_pos_text = ds_pos['sentence']

filename = "/work3/s174498/sst2_dataset/negative"
ds_neg = load_from_disk(filename)
ds_neg_text = ds_neg['sentence']

pos = [ds_pos_text[i] for i in list(np.random.choice(len(ds_pos_text),N))]
neg = [ds_neg_text[i] for i in list(np.random.choice(len(ds_neg_text),N))]

# Concept data 
# load hate
datadir = '/work3/s174498/concept_random_dataset/'
filename = 'tweet_hate/test'
ds_hate = load_from_disk(datadir + filename)
ds_hate = ds_hate['text']
hate = [ds_hate[i] for i in list(np.random.choice(len(ds_hate),M))]

# load woman 
filefolder = 'wikipedia_20220301/gender_concepts/'
filename = 'Woman'
ds_woman = load_from_disk(datadir +filefolder + filename)
ds_woman = ds_woman['text_list']
woman = [ds_woman[i] for i in list(np.random.choice(len(ds_woman),M))]

# load trans
filename = 'Transsexual'
ds_trans = load_from_disk(datadir +filefolder + filename)
ds_trans = ds_trans['text_list']
trans = [ds_trans[i] for i in list(np.random.choice(len(ds_trans),M))]

# load intersex
filename = 'Intersex'
ds_inter = load_from_disk(datadir +filefolder + filename)
ds_inter = ds_inter['text_list']
inter = [ds_inter[i] for i in list(np.random.choice(len(ds_inter),M))]


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

layers = ['roberta.encoder.layer.0.output.dense',
        'roberta.encoder.layer.1.output.dense',
        'roberta.encoder.layer.2.output.dense',
        'roberta.encoder.layer.3.output.dense',
        'roberta.encoder.layer.4.output.dense',
        'roberta.encoder.layer.5.output.dense',
        'roberta.encoder.layer.6.output.dense',
        'roberta.encoder.layer.7.output.dense',
        'roberta.encoder.layer.8.output.dense',
        'roberta.encoder.layer.9.output.dense']
        #'roberta.encoder.layer.10.output.dense',
        #'roberta.encoder.layer.11.output.dense']


if target_name == 'negative':
    target_data = neg
elif target_name == 'positive':
    target_data = pos
else:
    print('wrong target data name')


# TCAV data 
save_tcav = {}
save_tcav[target_name] = {concepts[0]:{layers[0] :{'TCAV':0 ,'acc':0}}, 'random':{layers[0]:{'TCAV':0}}}
for concept_name in concepts:
    if concept_name == 'hate':
        concept_data = hate #
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    elif concept_name == 'news':
        concept_data = news #
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    elif concept_name == 'Woman':
        concept_data = woman
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    elif concept_name == 'Intersex':
        concept_data = inter
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    elif concept_name == 'Transsexual':
        concept_data = trans
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    else:
        print('missing concet data name')

    for nr, layer in enumerate(layers):
        print(layer)
        print('TCAV for layer:', nr)
        _,_,TCAV, acc, _,TCAV_random = get_preds_tcavs(classifier = 'linear',model_layer=layer,layer_nr =nr,
                                        target_text = target_data, desired_class=target_nr,
                                        counter_set = 'wikipedia_split',
                                        concept_text = concept_data, 
                                        num_runs=num_random_set)
        save_tcav[target_name][concept_name][layer] = {'TCAV':TCAV, 'acc':acc}
        save_tcav[target_name]['random'][layer] = {'TCAV':TCAV_random}

# saving the file 
PATH =  f"/work3/s174498/nlp_tcav_results/{FILE_NAME}.pkl"
f = open(PATH ,"wb")
pickle.dump(save_tcav, f)
f.close()

print('FINISH')    

"""
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
"""