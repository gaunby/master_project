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
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split


random.seed(100)

PATH_TO_Data = '/work3/s174498/concept_random_dataset/'
#PATH_TO_Model = '/work3/s174498/final/linear_head/checkpoint-1500' #'/zhome/94/5/127021/speciale/master_project/src/models/hate_tcav/models/'

#with open(PATH_TO_Data+'data/random_stopword_tweets.txt','r') as f_:
#  random_examples= f_.read().split('\n\n')

#random_concepts = random_examples[-100:]
# load
random_data = load_from_disk(PATH_TO_Data + 'wikipedia_split')
random_text = random_data['complex_sentence'] # number of obs. = 989944
# random_concepts = [random_examples[i] for i in list(np.random.choice(len(random_examples),200))]


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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

def train_lm(lm, x, y):
  
  x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.33, stratify=y )
  
  lm.fit(x_train, y_train)
  y_pred = lm.predict(x_test)

  # return the fraction of correctly classified samples.
  acc = metrics.accuracy_score( y_test, y_pred, normalize = True)

  return lm, acc


def compute_cavs(model, tokenizer, concept_text, random_rep, num_runs=500, random_run = False):
  #calculates CAVs
  # num_runs: should be num_random_set
  cavs = []
  acc = []
  N = int(len(random_rep)/num_runs)
  print('CAVS:',N)

  if not random_run:
    class_concept = get_reps(model,tokenizer,concept_text)
    N = len(class_concept)
  
  if len(random_rep) < num_runs*N:
    print('Incorrect number of random samples.\nNeed',num_runs*N, 'but have',len(random_rep))
    return

  labels2text = {}
  labels2text[0] = 'concept'
  labels2text[1] = 'random'
  for i in range(num_runs-1):
    lm = linear_model.SGDClassifier(
      alpha=0.01,
      max_iter=5000,
      tol=1e-3,
      )
    class_random = random_rep[i*N:(i+1)*N]
    x = []
    labels = []
    if random_run:
      j = i
      random.seed(j)
      random_sample = random.randint(0, num_runs-1)
      while random_sample == i:
        j += 1
        random.seed(j)
        random_sample = random.randint(0, num_runs-1)
      class_concept = random_rep[random_sample*N : (random_sample+1)*N]

    x.extend(class_concept)
    labels.extend([0]*N)
    x.extend(class_random)
    labels.extend([1]*N)
    
    x = np.array(x)
    labels = np.array(labels)
    
    lm, acc_ = train_lm(lm, x, labels)

    # what they do in TCAV Been Kim : 
    # cavs.extend([-1 * lm.coef_[0],lm.coef_[0]])
    cavs.extend([lm.coef_[0]])
    acc.extend([acc_])
    
  # cavs[4] is cav-run number 4
  # cavs: list of arrays (the arrays are a list)
  return cavs, acc

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
  
  attention_mask = input['attention_mask'].to(device)
  
  logits, _, _ = model(input_ids, attention_mask=attention_mask)
  
  logits[0, desired_class].backward()#desired_class].backward() # must be a specific class
  
  grad = model.grad_representation # differs with input sample
  
  grad = grad[0][0].cpu().numpy()#grad.sum(dim=0).squeeze().cpu().detach().numpy()
  
  return logits,grad

def get_preds_tcavs(classifier = 'linear',model_layer = "roberta.encoder.layer.11.output.dense", layer_nr = '11',target_text = random_text,desired_class = 0,counter_set = 'wikipedia_split',concept_text = random_text, num_runs = 10):
  #returns logits, sensitivies and tcav score
  num_random_set = num_runs #num_random_set
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
  
  if len(concept_text) < 100:
    print('Too few concept text examples. Must be greater than 100')
    return

  if counter_set=='wikipedia_split':
    
    num_ex_in_set = len(concept_text)
    Data = counter_set #'wikipedia_split'
    
    file_name =  f'tensor_{Data}_on_{layer_nr}_layer_{num_random_set}_sets_with_{num_ex_in_set}' # f'tensor_{Data}_on_{layer_nr}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
    file_random = PATH_TO_Data + '/'+ Data + '/' + file_name + '.pt'

    if os.path.exists(file_random):
      random_rep = torch.load(file_random)
    else:
      print('Counter part does not have a representation for this model layer or does not have the correct size.\nCreate by running: embedding_layer_rep.py')
      return

  else:
    print('Counter part does not have a representation for this random dataset\nCreate by running: embedding_layer_rep.py')
    return

  print('calculating concept cavs...')
  model.to(device)
  concept_cavs, acc = compute_cavs(model,tokenizer, concept_text, random_rep, num_runs=num_runs)
  # CAVS Random
  PATH_random_cav = PATH_TO_Data+'cavs/'+classifier+ '_classifier_on_layer_' + str(layer_nr)+'_random.pkl'
  if os.path.exists(PATH_random_cav):
    print('cavs random are saved.')
    with open(PATH_random_cav,'rb') as handle:
      data = pickle.load(handle)
    cav_random = data['cavs']
    acc_random = data['acc']
    print('number of cavs',len(cav_random))
  else:
    print('calculating random cavs...')
    cav_random = []
    acc_random = []
    cav_random, acc_random =  compute_cavs(model,tokenizer, concept_text, random_rep, num_runs=num_runs, random_run = True)
    cav_random.append(cav_random)
    acc_random.append(acc_random)
    cav_data ={'cavs':cav_random, 'acc':acc_random}
    with open(PATH_random_cav, 'wb') as handle:
      pickle.dump(cav_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # Grads
  PATH_grad = '/work3/s174498/sst2_dataset/grads_logits/'+classifier+'_class_'+str(desired_class)+'_layer_'+str(layer_nr)+'.pkl'
  if os.path.exists(PATH_grad):
    print('logits and grads are saved.')
    with open(PATH_grad,'rb') as handle:
      data = pickle.load(handle)
    grads = data['grads']
    logits = data['logits']
  else:
    print('>>> calculating logits and grads...')
    logits = []
    grads = []
    for sample in target_text:
      logit,grad = get_logits_grad(model, tokenizer, sample, desired_class)

      grads.append(grad)
      logits.append(logit)
      data ={'grads':grads,
            'logits':logits}
    with open(PATH_grad, 'wb') as handle:
      pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
  sensitivities_concept = [] 

  for cav in concept_cavs:
    sensitivities_concept.append([np.dot(grad, (-1)*cav) for grad in grads])

  sensitivities_concept = np.array(sensitivities_concept) # each row is all grads on one cav 
  
  tcavs_concept = []
  
  for i in range(num_runs-1):
    tcavs_concept.append(len([s for s in sensitivities_concept[i,:] if s>0])/len(target_text))
  
  print('Accuracy over all:')
  print(np.mean(acc))
  print('TCAV score for the concept: ')
  print(np.mean(tcavs_concept),np.std(tcavs_concept)) 


  sensitivities_random = [] 
  
  for cav in cav_random[:-1]:
    #print(cav)
    #print('cav len',len(cav))
    sensitivities_random.append([np.dot(grad, (-1)*cav) for grad in grads])

  sensitivities_random = np.array(sensitivities_random) # each row is all grads on one cav 
  
  tcavs_random = []
  
  for i in range(num_runs-1):
    tcavs_random.append(len([s for s in sensitivities_random[i,:] if s>0])/len(target_text))
  
  print('Accuracy over all:')
  #print(np.mean(acc_random))
  print('TCAV score for the concept: ')
  print(np.mean(tcavs_random),np.std(tcavs_random))   
  return logits, sensitivities_concept, tcavs_concept, acc, sensitivities_random, tcavs_random