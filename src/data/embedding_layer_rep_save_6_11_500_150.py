import torch 
from datasets import load_from_disk
import sys
sys.path.insert(0,'/zhome/94/5/127021/speciale/master_project')
from src.data.embedding_layer_rep import create_embedding


PATH_TO_Data = '/work3/s174498/concept_random_dataset/'
Data = 'wikipedia_split'
random_data = load_from_disk(PATH_TO_Data + Data)
random_text = random_data['complex_sentence']


data = random_text
classifier = 'linear'
num_random_set = 500
num_ex_in_set = 150
print('start')
# layer 11
#model_layer = "roberta.encoder.layer.11.output.dense"
#model_layer_num = '11'

#random_rep = create_embedding(random_text, classifier, model_layer, num_random_set= num_random_set, num_ex_in_set= num_ex_in_set )

#name = f'tensor_{Data}_on_{model_layer_num}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
#file = PATH_TO_Data + Data + '/' + name + '.pt'
#torch.save(random_rep, file)

# layer 11
model_layer = "roberta.encoder.layer.11.output.dense"
model_layer_num = '11'

random_rep = create_embedding(random_text, classifier, model_layer, num_random_set= num_random_set, num_ex_in_set= num_ex_in_set )
print('embedding created ', model_layer_num)
name = f'tensor_{Data}_on_{model_layer_num}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
file = PATH_TO_Data + Data + '/' + name + '.pt'
torch.save(random_rep, file)
random_rep = 0
print('save embedding ' , model_layer_num)

# layer 10
model_layer = "roberta.encoder.layer.10.output.dense"
model_layer_num = '10'

random_rep = create_embedding(random_text, classifier, model_layer, num_random_set= num_random_set, num_ex_in_set= num_ex_in_set )
print('embedding created ', model_layer_num)
name = f'tensor_{Data}_on_{model_layer_num}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
file = PATH_TO_Data + Data + '/' + name + '.pt'
torch.save(random_rep, file)
random_rep = 0
print('save embedding ' , model_layer_num)

# layer 9
model_layer = "roberta.encoder.layer.9.output.dense"
model_layer_num = '9'

random_rep = create_embedding(random_text, classifier, model_layer, num_random_set= num_random_set, num_ex_in_set= num_ex_in_set )
print('embedding created ', model_layer_num)
name = f'tensor_{Data}_on_{model_layer_num}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
file = PATH_TO_Data + Data + '/' + name + '.pt'
torch.save(random_rep, file)
random_rep = 0
print('save embedding ' , model_layer_num)

# layer 8
model_layer = "roberta.encoder.layer.8.output.dense"
model_layer_num = '8'
print('start', model_layer_num)
random_rep = create_embedding(random_text, classifier, model_layer, num_random_set= num_random_set, num_ex_in_set= num_ex_in_set )

name = f'tensor_{Data}_on_{model_layer_num}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
file = PATH_TO_Data + Data + '/' + name + '.pt'
print('save', model_layer_num)
torch.save(random_rep, file)
random_rep = 0

# layer 7
model_layer = "roberta.encoder.layer.7.output.dense"
model_layer_num = '7'
print('start', model_layer_num)
random_rep = create_embedding(random_text, classifier, model_layer, num_random_set= num_random_set, num_ex_in_set= num_ex_in_set )

name = f'tensor_{Data}_on_{model_layer_num}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
file = PATH_TO_Data + Data + '/' + name + '.pt'
print('save', model_layer_num)
torch.save(random_rep, file)
random_rep = 0

# layer 6
model_layer = "roberta.encoder.layer.6.output.dense"
model_layer_num = '6'
print('start', model_layer_num)
random_rep = create_embedding(random_text, classifier, model_layer, num_random_set= num_random_set, num_ex_in_set= num_ex_in_set )

name = f'tensor_{Data}_on_{model_layer_num}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
file = PATH_TO_Data + Data + '/' + name + '.pt'
print('save', model_layer_num)
torch.save(random_rep, file)
random_rep = 0
"""
# layer 5
model_layer = "roberta.encoder.layer.5.output.dense"
model_layer_num = '5'
print('start', model_layer_num)
random_rep = create_embedding(random_text, classifier, model_layer, num_random_set= num_random_set, num_ex_in_set= num_ex_in_set )

name = f'tensor_{Data}_on_{model_layer_num}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
file = PATH_TO_Data + Data + '/' + name + '.pt'
print('save', model_layer_num)
torch.save(random_rep, file)
random_rep = 0

# layer 4
model_layer = "roberta.encoder.layer.4.output.dense"
model_layer_num = '4'
print('start', model_layer_num)
random_rep = create_embedding(random_text, classifier, model_layer, num_random_set= num_random_set, num_ex_in_set= num_ex_in_set )

name = f'tensor_{Data}_on_{model_layer_num}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
file = PATH_TO_Data + Data + '/' + name + '.pt'
print('save', model_layer_num)
torch.save(random_rep, file)
random_rep = 0

# layer 3
model_layer = "roberta.encoder.layer.3.output.dense"
model_layer_num = '3'
print('start', model_layer_num)
random_rep = create_embedding(random_text, classifier, model_layer, num_random_set= num_random_set, num_ex_in_set= num_ex_in_set )

name = f'tensor_{Data}_on_{model_layer_num}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
file = PATH_TO_Data + Data + '/' + name + '.pt'
print('save', model_layer_num)
torch.save(random_rep, file)
random_rep = 0

# layer 2
model_layer = "roberta.encoder.layer.2.output.dense"
model_layer_num = '2'
print('start', model_layer_num)
random_rep = create_embedding(random_text, classifier, model_layer, num_random_set= num_random_set, num_ex_in_set= num_ex_in_set )

name = f'tensor_{Data}_on_{model_layer_num}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
file = PATH_TO_Data + Data + '/' + name + '.pt'
print('save', model_layer_num)
torch.save(random_rep, file)
random_rep = 0

# layer 1
model_layer = "roberta.encoder.layer.1.output.dense"
model_layer_num = '1'
print('start', model_layer_num)
random_rep = create_embedding(random_text, classifier, model_layer, num_random_set= num_random_set, num_ex_in_set= num_ex_in_set )

name = f'tensor_{Data}_on_{model_layer_num}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
file = PATH_TO_Data + Data + '/' + name + '.pt'
print('save', model_layer_num)
torch.save(random_rep, file)
random_rep = 0

# layer 0
model_layer = "roberta.encoder.layer.0.output.dense"
model_layer_num = '0'
print('start', model_layer_num)
random_rep = create_embedding(random_text, classifier, model_layer, num_random_set= num_random_set, num_ex_in_set= num_ex_in_set )

name = f'tensor_{Data}_on_{model_layer_num}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
file = PATH_TO_Data + Data + '/' + name + '.pt'
print('save', model_layer_num)
torch.save(random_rep, file)
random_rep = 0
"""
print('end')