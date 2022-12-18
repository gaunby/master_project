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


datadir = '/work3/s174498/concept_random_dataset/wikipedia_20220301/gender_concepts/woman_female'
ds_woman = load_from_disk(datadir)
ds_woman = ds_woman.remove_columns(['title'])
ds_woman = ds_woman.rename_column('text_list','sentence')
ds_woman = ds_woman.add_column('label',[0]*len(ds_woman))
ds_woman = ds_woman.add_column('idx',list(range(len(ds_woman))))

datadir = '/work3/s174498/concept_random_dataset/wikipedia_split'
ds_random = load_from_disk(datadir)
ds_random = ds_random.remove_columns(['simple_sentence_1','simple_sentence_2'])
ds_random = ds_random.rename_column('complex_sentence','sentence')
#ds_random = ds_random.add_column('label',[0]*len(ds_random))
ds_random = ds_random.add_column('idx',list(range(0,len(ds_random))))
#ds_random2 = ds_random.add_column('idx',list(range(0,len(ds_random))))
#ds_random = ds_random.filter(lambda example, idx: idx <200000, with_indices=True)
random.seed(10)
ds_random_1 = ds_random.filter(lambda example, idx: idx in random.sample(range(0,len(ds_random)), len(ds_woman)), with_indices=True)
print(random.sample(range(0,len(ds_random))), len(ds_woman))
random.seed(12)
ds_random_2 = ds_random.filter(lambda example, idx: idx in random.sample(range(2,len(ds_random)), len(ds_woman)), with_indices=True)
print(random.sample(range(2,len(ds_random))), len(ds_woman))
ds_random_1 = ds_random_1.add_column('label',[1]*len(ds_random_1))
ds_random_2 = ds_random_2.add_column('label',[1]*len(ds_random_2))
print(len(ds_random_1))
print(len(ds_random_2))

ds = concatenate_datasets([ds_woman, ds_random_1,ds_random_2])
with open(f'/work3/s174498/roberta_files/data_set_woman_random_1_2.pickle', 'wb') as handle:
    pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

tokenized_test = ds.map(preprocess_function, batched=True)

trainer = Trainer(
    model=model,                        
    tokenizer=tokenizer
)

output = trainer.predict(tokenized_test)
with open(f'/work3/s174498/roberta_files/output_roberta_linear_woman_random_1_2.pickle', 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

# avg. runtime 2 min