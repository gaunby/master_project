import numpy as np
from datasets import load_from_disk
import pandas as pd
import pickle
import random
import sys
sys.path.insert(0,'/zhome/94/5/127021/speciale/master_project')
from src.models.tcav.TCAV import get_preds_tcavs

random.seed(1001)
np.random.seed(1001)

#############################################
######## SET ALL PARAMETERS HERE ############
#############################################

FILE_NAME = 'negative_sport' # name of saved file 
N = 300 # number of target examples 
M = 150 # number of concept examples

DROP_OUT = True
COUNTER_SET = 'wikipedia_split' #  'tweet_random' # 

num_random_set = 500 # number of runs/random folders

concepts = ['Acrobatic sports',
 'Air sports',
 'Aquatic and paddle sports',
 'Archery',
 'Athletics',
 'Bat and ball games',
 'Board game',
 'Boardsport',
 'Card game',
 'Catching games',
 'Climbing',
 'Combat sports',
 'Cycling',
 'Dog sports',
 'Electronic sports',
 'Equestrian sports',
 'Esports',
 'Fishing',
 'Flying disc sports',
 'Gymnastics',
 'Hunting',
 'Ice sports',
 'Invasion games',
 'Kite sports',
 'Marker sports',
 'Mixed discipline',
 'Motersport',
 'Net and wall games',
 'Orienteering family',
 'Other',
 'Other mind sports',
 'Overlapping sports',
 'Parkour Freerunning',
 'Remote control',
 'Rodeo',
 'Running',
 'Sailing',
 'Shooting sports',
 'Skating sports',
 'Snow sports',
 'Speedcubing',
 'Stacking',
 'Street sports',
 'Strength sports',
 'Table sports',
 'Tag game',
 'Target sport',
 'Walking',
 'Weightlifting']
target_nr = 0
target_name = 'negative'

############################################
############################################

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
# load woman 
filefolder = 'wikipedia_20220301/gender_concepts/'
filename = 'woman_female'
ds_woman = load_from_disk(datadir +filefolder + filename)
ds_woman = ds_woman['text_list']
woman = [ds_woman[i] for i in list(np.random.choice(len(ds_woman),M))]



if DROP_OUT:
    layers = ['roberta.encoder.layer.0.output.dropout',
             'roberta.encoder.layer.1.output.dropout',
             'roberta.encoder.layer.2.output.dropout',
             'roberta.encoder.layer.3.output.dropout',
             'roberta.encoder.layer.4.output.dropout',
             'roberta.encoder.layer.5.output.dropout',
             'roberta.encoder.layer.6.output.dropout',
             'roberta.encoder.layer.7.output.dropout',
             'roberta.encoder.layer.8.output.dropout',
             'roberta.encoder.layer.9.output.dropout',
             'roberta.encoder.layer.10.output.dropout',
             'roberta.encoder.layer.11.output.dropout'
            ]

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
    elif concept_name == 'offensive':
        concept_data = offen #
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    elif concept_name == 'irony':
        concept_data = irony #
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    elif concept_name == 'news':
        concept_data = ag_news #
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    elif concept_name == 'sport':
        concept_data = ag_sport #
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    elif concept_name == 'business':
        concept_data = ag_buss #
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    elif concept_name == 'world':
        concept_data = ag_world #
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    elif concept_name == 'science':
        concept_data = ag_sci #
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    elif concept_name == 'gender':
        concept_data = gender
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    elif concept_name == 'woman':
        concept_data = woman
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    elif concept_name == 'man':
        concept_data = man
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    elif concept_name == 'intersex':
        concept_data = inter
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    elif concept_name == 'transsexual':
        concept_data = trans
        save_tcav[target_name][concept_name] = {layers[0] :{'TCAV':0 ,'acc':0}}
    else:
        print('missing concet data name')

    for nr, layer in enumerate(layers):
        print(layer)
        print('TCAV for layer:', nr)
        _,sens,TCAV, acc, sens_random,TCAV_random, acc_random = get_preds_tcavs(classifier = 'linear',model_layer=layer,layer_nr =nr,
                                        target_text = target_data, desired_class=target_nr,
                                        counter_set = COUNTER_SET,
                                        concept_text = concept_data, concept_name= concept_name,
                                        num_runs=num_random_set,
                                        dropout=DROP_OUT)
        save_tcav[target_name][concept_name][layer] = {'TCAV':TCAV, 'acc':acc, 'sensitivity':sens}
        save_tcav[target_name]['random'][layer] = {'TCAV':TCAV_random,'acc':acc_random,'sensitivities':sens_random}

# saving the file 
PATH =  f"/work3/s174498/nlp_tcav_results/{FILE_NAME}.pkl"
f = open(PATH ,"wb")
pickle.dump(save_tcav, f)
f.close()

print('FINISH')    