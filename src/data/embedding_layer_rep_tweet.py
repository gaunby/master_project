import sys
from torch.utils.data.dataloader import DataLoader
import torch
from transformers import RobertaTokenizer
from datasets import load_from_disk
import numpy as np
import random
#sys.path.insert(0, '/zhome/a6/6/127219/Speciale/master_project')
sys.path.insert(0, '/zhome/94/5/127021/speciale/master_project')
from src.models.tcav.Roberta_model_data import RobertaClassifier,ToxicityDataset

random.seed(1234)

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

PATH_TO_Data = '/work3/s174498/concept_random_dataset/'
#Data = 'wikipedia_split'

def get_dataloader(X, y, tokenizer, batch_size):
  assert len(X) == len(y)
  encodings = tokenizer(X, truncation=True, padding=True, return_tensors="pt")
  dataset = ToxicityDataset(encodings, y)
  dataloader = DataLoader(dataset, batch_size=batch_size)
  return dataloader

def get_reps(model,tokenizer, concept_examples):
  #returns roberta representations of [CLS]   
  
  batch_size = 16 # to loop over 
  concept_labels = torch.zeros([len(concept_examples)]) #fake labels
  
  concept_repres = []
  concept_dataloader = get_dataloader(concept_examples,concept_labels,tokenizer,batch_size)
  print('       get data loader')
  with torch.no_grad():
    for i_batch, batch in enumerate(concept_dataloader):
      input_ids = batch['input_ids']#.to(device)
      attention_mask = batch['attention_mask']#.to(device)
      _, _, representation = model(input_ids, attention_mask=attention_mask)
      concept_repres.append(representation[:,0,:])
      
  concept_repres = torch.cat(concept_repres, dim=0).cpu().detach().numpy()
  
  return concept_repres


def create_embedding(random_text, classifier = 'linear',model_layer = "roberta.encoder.layer.11.output.dense", num_random_set = 10, num_ex_in_set = 150 ):
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
  
  model = RobertaClassifier(model_type = classifier, model_layer = model_layer)
  print('   model loaded')
  #N = num_ex_in_set
  #M = num_random_set

  #random_examples = random_text # [random_text[i] for i in list(np.random.choice(len(random_text),N*M))]
  print('   random examples')
  random_repres = get_reps(model,tokenizer,random_text)#random_examples)
  
  return random_repres

