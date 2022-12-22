import pickle 
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer
from datasets import load_from_disk 
import random
from datasets import concatenate_datasets

import sys
sys.path.insert(0, '/zhome/a6/6/127219/Speciale/master_project')
from src.visualization.tsne_visual import visualize_layerwise_embeddings#, visualize_one_layer
from src.models.transformers_modeling_roberta import RobertaForSequenceClassification_Linear, RobertaForSequenceClassification_Original

"""


"""

checkpoint = "/work3/s174498/final/linear_head/checkpoint-1500"

tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
tokenizer.model_max_len=512
model = RobertaForSequenceClassification_Linear.from_pretrained(checkpoint, output_hidden_states = True, return_dict = True)


def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)


#datadir = '/work3/s174498/concept_random_dataset/wikipedia_20220301/gender_concepts/woman_female'
#ds = load_from_disk(datadir)
#ds = ds.remove_columns(['title'])
#ds = ds.rename_column('text_list','sentence')
#ds = ds.add_column('label',[0]*len(ds))
#ds = ds.add_column('idx',list(range(len(ds))))
datadir = '/work3/s174498/concept_random_dataset/tweet_hate/test'
ds = load_from_disk(datadir)
ds = ds.filter(lambda ds: ds['label'] == 1)
ds = ds.add_column('idx',list(range(len(ds))))
ds = ds.remove_columns(['label'])
ds = ds.rename_column('text','sentence')
ds = ds.add_column('label',[0]*len(ds))

"""
datadir = '/work3/s174498/concept_random_dataset/wikipedia_split'
ds_random = load_from_disk(datadir)
ds_random = ds_random.remove_columns(['simple_sentence_1','simple_sentence_2'])
ds_random = ds_random.rename_column('complex_sentence','sentence')
ds_random = ds_random.add_column('idx',list(range(ds(len),len(ds_random)+ds(len))))
"""
datadir = '/work3/s174498/concept_random_dataset/tweet_random'
ds_random = load_from_disk(datadir)
ds_random = ds_random.remove_columns(['title'])
ds_random = ds_random.rename_column('text_list','sentence')
ds_random = ds_random.add_column('idx',list(range(len(ds),len(ds_random)+len(ds))))


random.seed(10)
ds_random_1 = ds_random.filter(lambda example, idx: idx in random.sample(range(len(ds),len(ds_random)+len(ds)), len(ds)), with_indices=True)

random.seed(12)
ds_random_2 = ds_random.filter(lambda example, idx: idx in random.sample(range(len(ds)+2,len(ds_random)+len(ds)), len(ds)), with_indices=True)

ds_random_1 = ds_random_1.add_column('label',[1]*len(ds_random_1))
ds_random_2 = ds_random_2.add_column('label',[1]*len(ds_random_2))
print(len(ds_random_1))
print(len(ds_random_2))

ds = concatenate_datasets([ds, ds_random_1,ds_random_2])
with open(f'/work3/s174498/roberta_files/data_set_tweet_random_1_2.pickle', 'wb') as handle:
    pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

tokenized_test = ds.map(preprocess_function, batched=True)

trainer = Trainer(
    model=model,                        
    tokenizer=tokenizer
)

output = trainer.predict(tokenized_test)
with open(f'/work3/s174498/roberta_files/output_roberta_linear_tweet_random_1_2.pickle', 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

# avg. runtime 2 min