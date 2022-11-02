
import pickle
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_from_disk
from tsne_visual import visualize_layerwise_embeddings

# load data
datadir = '/work3/s174498/sst2_dataset/'

test_dataset = load_from_disk(datadir + 'test_dataset')
#train_dataset = load_from_disk(datadir + 'train_dataset')
#validation_dataset = load_from_disk(datadir + 'validation_dataset')


# load
checkpoint = '/work3/s174498/finetuning-sentiment-model-all-samples-test6/checkpoint-1000'

# tokenizer
tokenizer_checkpoint = RobertaTokenizer.from_pretrained(checkpoint) 
tokenizer_pretrained = RobertaTokenizer.from_pretrained('roberta-base')
# model
model = RobertaForSequenceClassification.from_pretrained(checkpoint,output_hidden_states = True,return_dict = True)# output_attentions = True, 
model_pretrained = RobertaForSequenceClassification.from_pretrained('roberta-base',output_hidden_states = True,return_dict = True)

# Prepare the text inputs for the model
def preprocess_function(examples):
    return tokenizer_checkpoint(examples["sentence"], truncation=True)

tokenized_test = test_dataset.map(preprocess_function, batched=True)

with open('/work3/s174498/roberta_files/prediction_test6.pickle', 'rb') as handle:
    output_finetune = pickle.load(handle)

with open('/work3/s174498/roberta_files/output_roberta_base.pickle', 'rb') as handle:
    output_pretrained = pickle.load(handle)

# global 
labels = output_finetune.label_ids

hidden_states = output_finetune.predictions[1][1:]
title = 'finetuned_results'
layers_to_visualize = [0,1,2,3,4,5,6,7,8,9,10,11]
init = 'pca'
perplexity = 10
visualize_layerwise_embeddings(hidden_states,labels,title,layers_to_visualize, perplexity,init)
perplexity = 30
visualize_layerwise_embeddings(hidden_states,labels,title,layers_to_visualize, perplexity,init)
perplexity = 50
visualize_layerwise_embeddings(hidden_states,labels,title,layers_to_visualize, perplexity,init)
init = 'random'
visualize_layerwise_embeddings(hidden_states,labels,title,layers_to_visualize, perplexity,init)
perplexity = 10
visualize_layerwise_embeddings(hidden_states,labels,title,layers_to_visualize, perplexity,init)

labels = output_pretrained.label_ids

hidden_states = output_pretrained.predictions[1][1:]
title = 'pretrained_results'
layers_to_visualize = [0,1,2,3,4,5,6,7,8,9,10,11]
init = 'pca'
perplexity = 10
visualize_layerwise_embeddings(hidden_states,labels,title,layers_to_visualize, perplexity,init)
perplexity = 30
visualize_layerwise_embeddings(hidden_states,labels,title,layers_to_visualize, perplexity,init)
perplexity = 50
visualize_layerwise_embeddings(hidden_states,labels,title,layers_to_visualize, perplexity,init)
init = 'random'
visualize_layerwise_embeddings(hidden_states,labels,title,layers_to_visualize, perplexity,init)
perplexity = 10
visualize_layerwise_embeddings(hidden_states,labels,title,layers_to_visualize, perplexity,init)