print('starting')
import torch 
from datasets import load_from_disk
import sys
sys.path.insert(0,'/zhome/94/5/127021/speciale/master_project')
from src.data.embedding_layer_rep import create_embedding


PATH_TO_Data = '/work3/s174498/concept_random_dataset/'
Data = 'wikipedia_split'
random_data = load_from_disk(PATH_TO_Data + Data)
random_text = random_data['complex_sentence']


# data = random_text
classifier = 'linear'
num_random_set = 500
num_ex_in_set = 150
print('num random sets:',num_random_set,'\nex in set:',num_ex_in_set)

# layer 11
#model_layer = "roberta.encoder.layer.11.output.dense"
#model_layer_num = '11'

#random_rep = create_embedding(random_text, classifier, model_layer, num_random_set= num_random_set, num_ex_in_set= num_ex_in_set )

#name = f'tensor_{Data}_on_{model_layer_num}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
#file = PATH_TO_Data + Data + '/' + name + '.pt'
#torch.save(random_rep, file)

# layer 10
model_layer = "roberta.encoder.layer.10.output.dense"
model_layer_num = '10'
print('create embedding...')
random_rep = create_embedding(random_text, classifier, model_layer, num_random_set= num_random_set, num_ex_in_set= num_ex_in_set )
print('embedding created - 10 ')
name = f'tensor_{Data}_on_{model_layer_num}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
file = PATH_TO_Data + Data + '/' + name + '.pt'
print('file path:', file)
torch.save(random_rep, file)

print('end')