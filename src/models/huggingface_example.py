
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaTokenizer, DataCollatorWithPadding, RobertaForSequenceClassification, TrainingArguments, Trainer

import numpy as np
from datasets import load_dataset, load_metric
import torch

import wandb
wandb.init(
    project="SST2_sentiment_analysis",
    name='finetune_roberta_test3',
    entity="speciale",
    dir="/work3/s174498/wandb",
)

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

dataset = load_dataset("sst2")

#small_train_dataset = dataset["train"].shuffle(seed=42).select([i for i in list(range(100))])
#small_val_dataset = dataset["validation"].shuffle(seed=42).select([i for i in list(range(50))])

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

tokenized_train = dataset['train'].map(preprocess_function, batched=True)
tokenized_val = dataset['validation'].map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = RobertaForSequenceClassification.from_pretrained('roberta-base')

validation_inputs = tokenized_val.remove_columns(['label', 'idx'])
validation_targets = [tokenized_val.features['label'].int2str(x) for x in tokenized_val['label']]

validation_logger = ValidationDataLogger(
    inputs = validation_inputs[:],
    targets = validation_targets
)

repo_name = "/work3/s174498/finetuning-sentiment-model-all-samples-test3"

args = TrainingArguments(
    report_to='wandb',                    # enable logging to W&B
    output_dir=repo_name,                 # set output directory
    overwrite_output_dir=True,
    evaluation_strategy='steps',          # check evaluation metrics on a given # of steps
    learning_rate=2e-05,                  # we can customize learning rate
    per_device_train_batch_size = 16, 
    per_device_eval_batch_size = 16,
    logging_steps=100,                    # we will log every x steps
    eval_steps=500,                       # we will perform evaluation every x steps
    save_total_limit=1,
    num_train_epochs=5,
    gradient_accumulation_steps = 16, 
    eval_accumulation_steps=1,            # report evaluation results after each step
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    lr_scheduler_type = "linear"
)

# automatically log model to W&B at the end
#WANDB_LOG_MODEL=True

accuracy_metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    prediction_labels = [tokenized_val.features['label'].int2str(x.item())
                         for x in predictions]
    
    # log predictions
    validation_logger.log_predictions(prediction_labels)

    # metrics from the datasets library have a compute method
    return accuracy_metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,                        # model to be trained
    args=args,                          # training args
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,                # for padding batched data
    data_collator=data_collator,            
    compute_metrics=compute_metrics     # for custom metrics
)

trainer.train()