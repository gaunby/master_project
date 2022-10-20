# Import required packages
from transformers import RobertaTokenizer, DataCollatorWithPadding, RobertaForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_dataset
from datasets import load_metric

import wandb
wandb.init(project="SST2_sentiment_analysis",
            name="roberta-base-finetune",
            entity="mmfogh")

# Capture a dictionary of hyperparameters with config


# Prepare the text inputs for the model
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

# Define the evaluation metrics 
def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}

# 1. Load data from Huggingface
from datasets import load_dataset
#data = load_dataset('tweet_eval', 'hate')
data = load_dataset('sst2')

# Create a smaller training dataset for faster training times
small_train_dataset = data["train"].shuffle(seed=42).select([i for i in list(range(100))])
small_val_dataset = data["validation"].shuffle(seed=42).select([i for i in list(range(50))])

# 2. Preprocess data
# Set Roberta tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_val = small_val_dataset.map(preprocess_function, batched=True)

# Use data_collector to convert our samples to PyTorch tensors and concatenate them with the correct amount of padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 3. Train model
# Define RoBERTa as our base model:
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

# Define a new Trainer with all the objects we constructed so far
repo_name = "/work3/s174498/finetuning-sentiment-model-100-samples"

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    seed = 42,
    num_train_epochs=2,
    weight_decay=0.01,
    #save_strategy="epoch",
    lr_scheduler_type = "linear",
    evaluation_strategy = "epoch",
    #load_best_model_at_end = True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    report_to = wandb,
    WANDB_LOG_MODEL = True
)

# Train the model
trainer.train()