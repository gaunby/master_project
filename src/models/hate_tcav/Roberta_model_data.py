import sys
import torch.nn as nn
#import numpy as np 
#from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
import torch
sys.path.insert(0, '/zhome/a6/6/127219/Speciale/master_project')
#sys.path.insert(0, '/zhome/94/5/127021/speciale/master_project')
from src.models.transformers_modeling_roberta import RobertaForSequenceClassification_Linear, RobertaForSequenceClassification_Original



class ToxicityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class RobertaClassifier(nn.Module):
  def __init__(self,  model_type, model_layer):
    super(RobertaClassifier, self).__init__()
    
    if model_type == 'linear':
      folder = '/work3/s174498/final/linear_head/checkpoint-1500'
      self.roberta_classifier = RobertaForSequenceClassification_Linear.from_pretrained(folder)
    elif model_type == 'original':
      folder = '/work3/s174498/final/original_head/checkpoint-500'
      self.roberta_classifier = RobertaForSequenceClassification_Original.from_pretrained(folder)
  
    self.grad_representation = None
    self.representation = None
    # using the representation of layer12 in the transformer
    for name, module in self.roberta_classifier.named_modules():
      # for loop runs through all layers of the model 
      if name == model_layer: #"roberta.encoder.layer.11.output.dense": #roberta representation "classifier.dense": #
        module.register_forward_hook(self.forward_hook_fn)
        module.register_full_backward_hook(self.backward_hook_fn)

    self.roberta_classifier.requires_grad_(True)
  
  def forward_hook_fn(self, module, input, output):  #gradient
    self.representation = output

  def backward_hook_fn(self, module, grad_input, grad_output):
    self.grad_representation = grad_output[0]

  def forward(self, input_ids: torch.Tensor, attention_mask:torch.tensor, labels=None):
    if labels is not None:
      loss, logits = self.roberta_classifier(input_ids=input_ids, attention_mask= attention_mask, labels=labels)
    else:
      out = self.roberta_classifier(input_ids=input_ids, attention_mask=attention_mask)
      logits = out[0]

    preds = torch.argmax(logits, dim=-1)  # (batch_size, )

    if labels is None:
      return logits, preds, self.representation
    else:
      loss = nn.functional.cross_entropy(logits, labels)
      return loss, logits, preds, self.representation

  def forward_from_representation(self, representation: torch.Tensor):
    #classifier = nn.Sequential(self.xlnet_classifier.sequence_summary, self.xlnet_classifier.logits_proj)
    logits = self.roberta_classifier.classifier(representation)  # (batch_size, num_labels)
    
    preds = torch.argmax(logits, dim=-1)  # (batch_size, )
    return logits,preds