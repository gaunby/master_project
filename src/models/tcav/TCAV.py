#by Isar Nejadgholi
import torch.nn as nn
import numpy as np
import os
import pickle
import torch
from transformers import RobertaTokenizer # Fast
from torch.utils.data.dataloader import DataLoader
from src.models.tcav.Roberta_model_data import RobertaClassifier,GetDataset
import random
from datasets import load_from_disk
import time
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split


random.seed(123456)
np.random.seed(1234)

PATH_TO_Data = '/work3/s174498/concept_random_dataset/'

# load
random_data = load_from_disk(PATH_TO_Data + 'wikipedia_split')
random_text = random_data['complex_sentence'] # 

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_dataloader(X, y, tokenizer, batch_size):
  assert len(X) == len(y)
  encodings = tokenizer(X, truncation=True, padding=True, return_tensors="pt")
  dataset = GetDataset(encodings, y)
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
      input_ids = batch['input_ids']#.to(device)
      attention_mask = batch['attention_mask']#.to(device)
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
  acc_train = metrics.accuracy_score( y_test, y_pred, normalize = True)

  return lm, acc_train


def compute_cavs(model, tokenizer, concept_text, random_rep, num_runs=500, random_run = False):
  #calculates CAVs
  # num_runs: should be num_random_set
  cavs = []
  acc_cav = []
  N = int(len(random_rep)/num_runs)
  print('Number ex (in each class) to compute CAVs on:',N)

  if not random_run:
    class_concept = get_reps(model,tokenizer,concept_text)
    N = len(class_concept)
  
  if len(random_rep) < num_runs*N:
    print('Incorrect number of random samples.\nNeed',num_runs*N, 'but have',len(random_rep))
    return

  labels2text = {}
  labels2text[0] = 'concept'
  labels2text[1] = 'random'
  for i in range(num_runs):
    lm = linear_model.SGDClassifier(
      alpha=0.01,
      max_iter=5000,
      tol=1e-3, # default 
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
    acc_cav.extend([acc_])
    
  # cavs[4] is cav-run number 4
  # cavs: list of arrays (the arrays are a list)
  return cavs, acc_cav


def get_logits_grad(model, tokenizer, sample, desired_class):
  #returns logits and gradients
  input = tokenizer(sample, truncation=True,padding=True, return_tensors="pt")
  model.zero_grad()
  input_ids = input['input_ids'] # idx from vocab
  
  attention_mask = input['attention_mask']
  
  logits, _, _ = model(input_ids, attention_mask=attention_mask)
  
  logits[0, desired_class].backward() # must be a specific class
  
  grad = model.grad_representation # differs with input sample
  
  grad = grad[0][0].cpu().numpy()
  
  return logits,grad

def get_probs_grad(model, tokenizer, sample, desired_class):
  #returns probs and gradients
  input = tokenizer(sample, truncation=True,padding=True, return_tensors="pt")
  model.zero_grad()
  input_ids = input['input_ids'] # idx from vocab
  
  attention_mask = input['attention_mask']
  
  probs, _, _ = model(input_ids, attention_mask=attention_mask)
  
  probs[0, desired_class].backward() # must be a specific class
  
  grad = model.grad_representation # differs with input sample
  
  grad = grad[0][0].cpu().numpy()
  
  return probs,grad


def get_preds_tcavs(classifier = 'linear',model_layer = "roberta.encoder.layer.11.output.dense", layer_nr = '11',target_text = random_text,desired_class = 0,counter_set = 'wikipedia_split',concept_text = random_text,concept_name = 'random', num_runs = 10, dropout = False):
  #returns logits, sensitivies and tcav score
  num_random_set = num_runs #num_random_set
  # load tokenizer 
  if classifier == 'linear':
    folder = '/work3/s174498/final/Prob_linear_head/checkpoint-2500' #'/work3/s174498/final/linear_head/checkpoint-1500'
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
    if dropout:
      file_name =  f'tensor_{Data}_on_{layer_nr}_layer_dropout_{num_random_set}_sets_with_{num_ex_in_set}' # f'tensor_{Data}_on_{layer_nr}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
      file_random = PATH_TO_Data +  Data + '/' + file_name + '.pt'
      random_rep = torch.load(file_random)
    elif os.path.exists(file_random):
      file_name =  f'tensor_{Data}_on_{layer_nr}_layer_{num_random_set}_sets_with_{num_ex_in_set}' # f'tensor_{Data}_on_{layer_nr}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
      file_random = PATH_TO_Data + Data + '/' + file_name + '.pt'
      random_rep = torch.load(file_random)
    else:
      print('Counter part does not have a representation for this model layer or does not have the correct size.\nCreate by running: embedding_layer_rep.py')
      return
  elif counter_set=='tweet_random':
    num_ex_in_set = len(concept_text)
    Data = counter_set 
    all_data = [1,2,3,4,5,9,10,11]
    if dropout:
      file_name =  f'tensor_{Data}_on_{layer_nr}_layer_dropout_{num_random_set}_sets_with_{num_ex_in_set}' # f'tensor_{Data}_on_{layer_nr}_layer_{num_random_set}_sets_with_{num_ex_in_set}'
      file_random = PATH_TO_Data +  Data + '/' + file_name + '.pt'
      random_rep = torch.load(file_random)
      if layer_nr in all_data:
        random_rep = [random_rep[i] for i in list(np.random.choice(len(random_rep),num_random_set*num_ex_in_set))]
    else:
      print('Counter part does not have a representation for this model layer or does not have the correct size.\nCreate by running: embedding_layer_rep.py')
      return
  else:
    print('Counter part does not have a representation for this random dataset\nCreate by running: embedding_layer_rep.py')
    return

  # CAVS concept
  if dropout:
    PATH_concept_cav = PATH_TO_Data+'cavs/concept/'+concept_name+'_'+classifier+ '_classifier_on_layer_dropout_' + str(layer_nr)+'_with_'+str(num_runs)+'random.pkl'
  else:
    PATH_concept_cav = PATH_TO_Data+'cavs/concept/'+concept_name+'_'+classifier+ '_classifier_on_layer_' + str(layer_nr)+'_with_'+str(num_runs)+'random.pkl'

  if os.path.exists(PATH_concept_cav):
    print('cavs concept are saved.')
    with open(PATH_concept_cav,'rb') as handle:
      data = pickle.load(handle)
    concept_cavs = data['cavs']
    acc = data['acc']
    print('number of concept cavs:',len(concept_cavs))
  else:
    print('calculating concept cavs...')
    concept_cavs = []
    acc = []
    cavs_out, acc_out =  compute_cavs(model,tokenizer, concept_text, random_rep, num_runs=num_runs)
    concept_cavs.extend(cavs_out)
    acc.extend(acc_out)
    cav_data ={'cavs':concept_cavs, 'acc':acc}
    with open(PATH_concept_cav, 'wb') as handle:
      pickle.dump(cav_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # CAVS Random
  if dropout:
    if counter_set == 'wikipedia_split':
      PATH_random_cav = PATH_TO_Data+'cavs/random/'+classifier+ '_classifier_on_layer_dropout_' + str(layer_nr)+'_with_'+str(num_runs)+'random.pkl'
    if counter_set == 'tweet_random':
      PATH_random_cav = PATH_TO_Data+'cavs/random/'+classifier+ '_classifier_on_layer_dropout_' + str(layer_nr)+'_with_'+str(num_runs)+'random_TWEET.pkl'
  else:
    PATH_random_cav = PATH_TO_Data+'cavs/random/'+classifier+ '_classifier_on_layer_' + str(layer_nr)+'_with_'+str(num_runs)+'random.pkl'

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
    cav_out, acc_out =  compute_cavs(model,tokenizer, concept_text, random_rep, num_runs=num_runs, random_run = True)
    cav_random.extend(cav_out)
    acc_random.extend(acc_out)
    cav_data ={'cavs':cav_random, 'acc':acc_random}
    with open(PATH_random_cav, 'wb') as handle:
      pickle.dump(cav_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # Grads
  if dropout:
    #PATH_grad = '/work3/s174498/sst2_dataset/grads_logits/'+classifier+'_class_'+str(desired_class)+'_layer_dropout_'+str(layer_nr)+'.pkl'
    PATH_grad = '/work3/s174498/sst2_dataset/grads_probs/'+classifier+'_class_'+str(desired_class)+'_layer_dropout_'+str(layer_nr)+'.pkl'
  else:
    PATH_grad = '/work3/s174498/sst2_dataset/grads_logits/'+classifier+'_class_'+str(desired_class)+'_layer_'+str(layer_nr)+'.pkl'

  if os.path.exists(PATH_grad) and torch.cuda.is_available():
    print('logits and grads are saved.')
    with open(PATH_grad,'rb') as handle:
      data = pickle.load(handle)
    grads = data['grads']
    probs = data['probs'] #data['logits']
  else:
    print('>>> calculating logits/probs and grads...')
    probs = []
    grads = []
    for sample in target_text:
      prob,grad = get_probs_grad(model, tokenizer, sample, desired_class) #get_logits_grad(model, tokenizer, sample, desired_class)

      grads.append(grad)
      probs.append(prob)
      data ={'grads':grads,
            'probs':probs}
    with open(PATH_grad, 'wb') as handle:
      pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
  sensitivities_concept = [] 
  print('number of concept cavs:',len(concept_cavs))
  for cav in concept_cavs:
    sensitivities_concept.append([np.dot(grad, (-1)*cav) for grad in grads])
  
  sensitivities_concept = np.array(sensitivities_concept) # each row is all grads on one cav 
  
  tcavs_concept = []
  print('sens shape:', sensitivities_concept.shape)
  for i in range(num_runs):
    tcavs_concept.append(len([s for s in sensitivities_concept[i,:] if s>0])/len(target_text))
  
  print('number tcavs concept:',len(tcavs_concept))
  print('Accuracy over all:')
  print(np.mean(acc))
  print('TCAV score for the concept: ')
  print(np.mean(tcavs_concept),np.std(tcavs_concept)) 


  sensitivities_random = [] 
  
  for cav in cav_random:
    sensitivities_random.append([np.dot(grad, (-1)*cav) for grad in grads])
  
  
  sensitivities_random = np.array(sensitivities_random) # each row is all grads on one cav 
  print('sen random shape:',sensitivities_random.shape)
  
  tcavs_random = []
  
  for i in range(num_runs):
    tcavs_random.append(len([s for s in sensitivities_random[i,:] if s>0])/len(target_text))
  
  print('number tcavs random:',len(tcavs_random))
  print('Accuracy over all:')
  print(np.mean(acc_random))
  print('TCAV score for the concept: ')
  print(np.mean(tcavs_random),np.std(tcavs_random))   
  return probs, sensitivities_concept, tcavs_concept, acc, sensitivities_random, tcavs_random, acc_random