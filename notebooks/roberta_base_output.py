import pickle 
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer
from datasets import load_from_disk 

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base',output_hidden_states = True,return_dict = True)


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
with open(f'/work3/s174498/roberta_files/output_roberta_base.pickle', 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

# avg. runtime 2 min