from datasets import load_dataset, Dataset
from more_itertools import roundrobin
import pyarrow as pa
import pandas as pd
import random 

# Import dataset from HuggingFace
dataset = load_dataset("sst2")

# validation set
validation_dataset = dataset['validation']

N = len(dataset['train']['idx'])
random_list = list(range(N))
random.Random(42).shuffle(random_list)
print('N:',N,'\nrandom list:', len(random_list))


# Sample test set from original train data
# Using the same distribution as seen in the validation set

dataset_test_idx_0 = []
dataset_test_sentence_0 = []
dataset_test_label_0 = []

dataset_test_idx_1 = []
dataset_test_sentence_1 = []
dataset_test_label_1 = []

for i in random_list[:2000]:
    if len(dataset_test_label_0) == 895 and len(dataset_test_label_1) == 926:
        break
    if (dataset["train"]["label"][i] == 0):
        if len(dataset_test_label_0) < 895:
            dataset_test_idx_0.append(dataset["train"]["idx"][i])
            dataset_test_sentence_0.append(dataset["train"]["sentence"][i])
            dataset_test_label_0.append(dataset["train"]["label"][i])
    if (dataset["train"]["label"][i] == 1):
        if len(dataset_test_label_1) < 926:
            dataset_test_idx_1.append(dataset["train"]["idx"][i])
            dataset_test_sentence_1.append(dataset["train"]["sentence"][i])
            dataset_test_label_1.append(dataset["train"]["label"][i])

# Gather lists containing labels 0 and 1
dataset_test_idx = list(roundrobin(dataset_test_idx_0,dataset_test_idx_1))
dataset_test_sentence = list(roundrobin(dataset_test_sentence_0,dataset_test_sentence_1))
dataset_test_label = list(roundrobin(dataset_test_label_0,dataset_test_label_1))

train_idx = list(set(dataset["train"]["idx"]) - set(dataset_test_idx))

dataset_train_label = [dataset["train"]["label"][i] for i in train_idx]
dataset_train_sentence = [dataset["train"]["sentence"][i] for i in train_idx] 
dataset_train_idx = train_idx


# lists as dictionary
# per class
test_0_df = pd.DataFrame({'idx': dataset_test_idx_0, 'sentence': dataset_test_sentence_0, 'label': dataset_test_label_0})
test_1_df = pd.DataFrame({'idx': dataset_test_idx_1, 'sentence': dataset_test_sentence_1, 'label': dataset_test_label_1})
# total
test_df = pd.DataFrame({'idx': dataset_test_idx, 'sentence': dataset_test_sentence, 'label': dataset_test_label})
train_df = pd.DataFrame({'idx': dataset_train_idx, 'sentence': dataset_train_sentence, 'label': dataset_train_label})

# convert to Huggingface dataset
test_0_dataset = Dataset(pa.Table.from_pandas(test_0_df))
test_1_dataset = Dataset(pa.Table.from_pandas(test_1_df))
test_dataset = Dataset(pa.Table.from_pandas(test_df))
train_dataset = Dataset(pa.Table.from_pandas(train_df))

# save datasets
test_0_dataset.save_to_disk('/work3/s174498/sst2_dataset/test_0_dataset')
test_1_dataset.save_to_disk('/work3/s174498/sst2_dataset/test_1_dataset')
train_dataset.save_to_disk('/work3/s174498/sst2_dataset/train_dataset_new')
validation_dataset.save_to_disk('/work3/s174498/sst2_dataset/validation_dataset_new')
test_dataset.save_to_disk('/work3/s174498/sst2_dataset/test_dataset_new')