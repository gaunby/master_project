import pickle 
#from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer
from transformers import RobertaTokenizer, Trainer
from datasets import load_from_disk 

import os
import sys
sys.path.insert(0, '/zhome/a6/6/127219/Speciale/master_project')
from src.models.transformers_modeling_roberta import RobertaForSequenceClassification_Linear, RobertaForSequenceClassification_Original

''''
checkpoint = "/work3/s174498/final/test_linear_head/checkpoint-1500"

tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
tokenizer.model_max_len=512
model = RobertaForSequenceClassification_Linear.from_pretrained(checkpoint, output_hidden_states = True, return_dict = True)


def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, max_length=512)


datadir = '/work3/s174498/sst2_dataset/'
test_dataset = load_from_disk(datadir + 'test_dataset')

tokenized_test = test_dataset.map(preprocess_function, batched=True)

trainer = Trainer(
    model=model,                        
    tokenizer=tokenizer
)

output = trainer.predict(tokenized_test)
with open(f'/work3/s174498/roberta_files/final_output_roberta_head_linear.pickle', 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''''

checkpoint = "/work3/s174498/final/test_original_head/checkpoint-1500"

tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
tokenizer.model_max_len=512
model = RobertaForSequenceClassification_Original.from_pretrained(checkpoint, output_hidden_states = True,return_dict = True)

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

datadir = '/work3/s174498/sst2_dataset/'
test_dataset = load_from_disk(datadir + 'test_dataset')

tokenized_test = test_dataset.map(preprocess_function, batched=True)

trainer = Trainer(
    model=model,                        
    tokenizer=tokenizer
)

output = trainer.predict(tokenized_test)
with open(f'/work3/s174498/roberta_files/checkoutput_final_output_roberta_head_nn.pickle', 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

# avg. runtime 5 min
