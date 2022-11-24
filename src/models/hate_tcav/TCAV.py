#by Isar Nejadgholi
import torch.nn as nn
import numpy as np
import os
import pickle
import torch
from transformers import RobertaTokenizer # Fast
from torch.utils.data.dataloader import DataLoader
from src.models.hate_tcav.Roberta_model_data import RobertaClassifier,ToxicityDataset
import random
from datasets import load_from_disk
import time

#random.seed(100)


PATH_TO_Data = '/work3/s174498/concept_random_dataset/'
#PATH_TO_Model = '/work3/s174498/final/linear_head/checkpoint-1500' #'/zhome/94/5/127021/speciale/master_project/src/models/hate_tcav/models/'

#with open(PATH_TO_Data+'data/random_stopword_tweets.txt','r') as f_:
#  random_examples= f_.read().split('\n\n')

#random_concepts = random_examples[-100:]
# load
# load 
random_data = load_from_disk(PATH_TO_Data + 'wikipedia_split')
random_text = random_data['complex_sentence'] # number of obs. = 989944
# random_concepts = [random_examples[i] for i in list(np.random.choice(len(random_examples),200))]


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

"""
model_folder_toxic = PATH_TO_Model#+'exp-Toxic-roberta/'
model_folder_Founta = PATH_TO_Model+'exp-Founta_original_roberta'
model_folder_EA = PATH_TO_Model+'exp-EA_2_class_roberta'
model_folder_CH = PATH_TO_Model+'CH_roberta'
model_folder_CH_explicit = PATH_TO_Model+'explicit_CH_roberta'
model_folder_toxic_explicit = PATH_TO_Model+'explicit_wiki_roberta'
"""

def get_dataloader(X, y, tokenizer, batch_size):
  assert len(X) == len(y)
  encodings = tokenizer(X, truncation=True, padding=True, return_tensors="pt")
  dataset = ToxicityDataset(encodings, y)
  dataloader = DataLoader(dataset, batch_size=batch_size)
  return dataloader

def get_reps(model,tokenizer, concept_examples):
  #returns roberta representations of [CLS]   
  
  batch_size = 8 # to loop over 
  concept_labels = torch.zeros([len(concept_examples)]) #fake labels
  
  concept_repres = []
  concept_dataloader = get_dataloader(concept_examples,concept_labels,tokenizer,batch_size)
  with torch.no_grad():
    for i_batch, batch in enumerate(concept_dataloader):
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      _, _, representation = model(input_ids, attention_mask=attention_mask)
      
      concept_repres.append(representation[:,0,:])
      
  concept_repres = torch.cat(concept_repres, dim=0).cpu().detach().numpy()
  
  return concept_repres

def compute_cavs(model, tokenizer, concept_text, random_rep, num_runs=10):
  #calculates CAVs
  cavs = []
  
  concept_rep = get_reps(model,tokenizer,concept_text)
  
  #####
  #####
  # build from here 



  #####
  #####
  for i in range(num_runs):
    concept_rep_ids = list(np.random.choice(range(len(concept_repres)), 30))
    concept_rep = [concept_repres[i] for i in concept_rep_ids]
    cavs.append(np.mean(concept_rep, axis = 0))
    
  # cavs: list of arrays (the arrays are a list)
  return cavs

# old function
def statistical_testing(model, tokenizer, concept_examples, num_runs=10): # old function
  #calculates CAVs
  cavs = []
  concept_repres = get_reps(model,tokenizer,concept_examples)
  for i in range(num_runs):
    concept_rep_ids = list(np.random.choice(range(len(concept_repres)), 30))
    concept_rep = [concept_repres[i] for i in concept_rep_ids]
    cavs.append(np.mean(concept_rep, axis = 0))
  return cavs

def get_logits_grad(model, tokenizer, sample, desired_class):
  #returns logits and gradients
  input = tokenizer(sample, truncation=True,padding=True, return_tensors="pt")
  model.zero_grad()
  input_ids = input['input_ids'].to(device) # idx from vocab
  
  #print('input', len(input))
  #print('>>> GET LOGITS GRAD input ids', input_ids)
  
  attention_mask = input['attention_mask'].to(device)
  logits, _, representation = model(input_ids, attention_mask=attention_mask)
  
  #print('logits',logits) # tror rigtig meget at logits er output 
  #print('rep', representation.shape)
  
  logits[0, desired_class].backward()
  #print('logits',logits)
  
  #print('cav shape',cav.shape)
  grad = model.grad_representation # differs with input sample

  #print('grad 1:', grad[0:3,0:3])
  #print('grad 2:', grad.shape)
  
  #print('first',grad.shape)
  grad = grad[0][0].cpu().numpy()#grad.sum(dim=0).squeeze().cpu().detach().numpy()
  return logits,grad

#def get_preds_tcavs(classifier = 'toxicity',desired_class = 1,examples_set = 'random',concept_examples = random_concepts, num_runs = 10):
def get_preds_tcavs(classifier = 'linear',model_layer = "roberta.encoder.layer.11.output.dense",desired_class = 1,counter_set = 'wikipedia_split',concept_text = random_text, num_runs = 10):
  #returns logits, sensitivies and tcav score
  num_random_set = 10
  # load tokenizer 
  if classifier == 'linear':
    folder = '/work3/s174498/final/linear_head/checkpoint-1500'
    tokenizer = RobertaTokenizer.from_pretrained(folder)
  elif classifier == 'original':
    folder = '/work3/s174498/final/original_head/checkpoint-500'
    tokenizer = RobertaTokenizer.from_pretrained(folder)
  else:
    print('model is unknown')
    return 
  
  model = RobertaClassifier(model_type = classifier, model_layer = model_layer )
  
  """
  if classifier=='toxicity':
    model = RobertaClassifier(model_folder_toxic)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_folder_toxic)
  elif classifier=='Founta':
    model = RobertaClassifier(model_folder_toxic)
    #print('>>> MODEL', model)
    #model = RobertaClassifier(model_folder_Founta)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_folder_toxic)
  elif classifier =='EA':
    model = RobertaClassifier(model_folder_toxic)#EA)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_folder_toxic)#EA)
  elif classifier =='CH':
    model = RobertaClassifier(model_folder_CH)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_folder_CH)
  elif classifier =='CH_exp':
    model = RobertaClassifier(model_folder_CH_explicit)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_folder_CH_explicit)  
  elif classifier =='wiki_exp':  
    model = RobertaClassifier(model_folder_toxic_explicit)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_folder_toxic_explicit)
  else:
    print('model is unknown')
    return 
  """
  if len(concept_text) < 100:
    print('Too few concept text examples. Must be greater than 100')
    return

  if counter_set=='wikipedia_split':
    num_ex_in_set = len(concept_text)
    Data = 'wikipedia_split'
    file_name = f'tensor_{Data}_on_{model_layer}_{num_random_set}_sets_with_{num_ex_in_set}'
    file_random = PATH_TO_Data + '/'+ Data + '/' + file_name + '.pt'
    random_rep = torch.load(file_random)
    #print('Number of random examples:', len(random_text))#
    #random_examples = [random_text[i][5:-1] for i in list(np.random.choice(len(random_text),N*num_random_set))]
  else:
    print('Counter part does not have a representation for this layer\nCreate by running: embedding_layer_rep.py')
    return

  print('calculating cavs...')
  model.to(device)
  concept_cavs = compute_cavs(model,tokenizer, concept_text, random_rep, num_runs=num_runs)
  #concept_cavs = statistical_testing(model,tokenizer, concept_examples, num_runs=num_runs)
  save = False
  """
  if os.path.exists('grads_logits/'+classifier+'_'+examples_set+'_'+str(desired_class)+'.pkl'):
    print('logits and grads are saved.')
    with open('grads_logits/'+classifier+'_'+examples_set+'_'+str(desired_class)+'.pkl','rb') as handle:
      data = pickle.load(handle)
    
    grads = data['grads']
    logits = data['logits']
    """
  if save:
    A = 0
  else:
    print('>>> calculating logits and grads...')
    logits = []
    grads = []
    for sample in examples:
      logit,grad = get_logits_grad(model, tokenizer, sample, desired_class)
      #print('>>> logit', logit)
      #print('>>> grad', grad.shape)
      grads.append(grad)
      logits.append(logit)
      data ={'grads':grads,
            'logits':logits}
    print(os.getcwd())
    with open('grads_logits/'+classifier+'_'+examples_set+'_'+str(desired_class)+'.pkl', 'wb') as handle:
      pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
   
  sensitivities = [] 
  print('>>> FOR LOOP GRADS', len(grads))
  for grad in grads:
    sensitivities.append([np.dot(grad, cav) for cav in concept_cavs])
  sensitivities = np.array(sensitivities)
  print('>>> sensitivet\n',sensitivities)
  print('>>> concetp cavs', len(concept_cavs))
  print(sensitivities.shape)
  print('examples length', len(examples))
  tcavs = []
  for i in range(num_runs):
    tcavs.append(len([s for s in sensitivities[:,i] if s>0])/len(examples))
   
  print('TCAV score for the concept: ')
  print(np.mean(tcavs),np.std(tcavs)) 
  
  return logits, sensitivities, tcavs
