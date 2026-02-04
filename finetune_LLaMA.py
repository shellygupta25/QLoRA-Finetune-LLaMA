import numpy as np
import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
import torch.nn.functional as F
from tqdm import tqdm
from peft import PeftModel

import os
import torch
import logging
os.environ["WANDB_DISABLED"] = "true"

logging.basicConfig(filename=f'finetune_LLaMA_hyperparameters_sequential_TT.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
df = pd.read_csv("pubmed_training_LLaMA_classification_head.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
################3 train test val split ######################3
train_end_point = int(df.shape[0]*0.70)
val_end_point = int(df.shape[0]*0.80)
train_df = df.iloc[:train_end_point,:]
val_df = df.iloc[train_end_point:val_end_point,:]
test_df = df.iloc[val_end_point:,:]
# Shuffle the dataframes to ensure randomness
train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)
val_df = val_df.sample(frac=1).reset_index(drop=True)
#print(train_df.shape, test_df.shape, val_df.shape)

# Converting pandas DataFrames into Hugging Face Dataset objects:
dataset_train = Dataset.from_pandas(train_df)
dataset_val = Dataset.from_pandas(val_df)
dataset_test = Dataset.from_pandas(test_df)

# Combine them into a single DatasetDict
dataset = DatasetDict({
'train': dataset_train,
'val': dataset_val,
'test': dataset_test
})

# Find class weights
class_weights=(1/train_df.label.value_counts(normalize=True).sort_index()).tolist()
class_weights=torch.tensor(class_weights)
class_weights=class_weights/class_weights.sum()

def model_train(r_h,lora_alpha_h,lora_dropout_h,epochs_h,batch_size_h,weight_decay_h,learning_rate_h,name_of_model):
  # load LLaMA model
  quantization_config = BitsAndBytesConfig(
  load_in_4bit = True,
  bnb_4bit_quant_type = 'nf4',
  bnb_4bit_use_double_quant = True,
  bnb_4bit_compute_dtype = torch.bfloat16
  )
  model_name = "./llama-3.1-8b"
  logger.info("loading Llama model")
  model = AutoModelForSequenceClassification.from_pretrained(
  model_name,
  quantization_config=quantization_config,
  num_labels=2,
  device_map='auto'
  )
  lora_config = LoraConfig(
  r = r_h,
  lora_alpha = lora_alpha_h,
  target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
  lora_dropout = lora_dropout_h,
  bias = 'none',
  task_type = 'SEQ_CLS'
  )
  model = prepare_model_for_kbit_training(model)
  model = get_peft_model(model, lora_config)
  model_name = "./llama-3.1-8b"
  tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
  tokenizer.pad_token_id = tokenizer.eos_token_id
  tokenizer.pad_token = tokenizer.eos_token
  model.config.pad_token_id = tokenizer.pad_token_id
  model.config.use_cache = False
  model.config.pretraining_tp = 1


  def get_metrics_result(test_df):
    y_test = test_df.label
    y_pred = test_df.predictions
    logger.info("\nClassification Report:\n")
    logger.info(classification_report(y_test, y_pred, digits = 4))
    logger.info(f"\nBalanced Accuracy Score: {balanced_accuracy_score(y_test, y_pred)}")
    logger.info(f"\nAccuracy Score: {accuracy_score(y_test, y_pred)}")
    return f1_score(y_test, y_pred, average='macro')


  def data_preprocesing(row):
    return tokenizer(row['input'], truncation=True, max_length=1024)

  tokenized_data = dataset.map(data_preprocesing, batched=True,
  remove_columns=['input'])
  tokenized_data.set_format("torch")
  collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
  
  def compute_metrics(evaluations):
    predictions, labels = evaluations
    predictions = np.argmax(predictions, axis=1)
    return {'balanced_accuracy' : balanced_accuracy_score(labels, predictions),
    'accuracy':accuracy_score(labels,predictions)}
    
  class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
      super().__init__(*args, **kwargs)
      if class_weights is not None:
        self.class_weights = torch.tensor(class_weights,
        dtype=torch.float32).to(self.args.device)
      else:
        self.class_weights = None
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
      labels = inputs.pop("labels").long()

      outputs = model(**inputs)
      logits = outputs.get("logits")

      if self.class_weights is not None:
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
      else:
        loss = F.cross_entropy(logits, labels)

      return (loss, outputs) if return_outputs else loss

  training_args = TrainingArguments(
  output_dir = 'sentiment_classification',
  learning_rate = learning_rate_h,
  per_device_train_batch_size = batch_size_h,
  per_device_eval_batch_size = batch_size_h,
  num_train_epochs = epochs_h,
  logging_steps=1,
  weight_decay = weight_decay_h,
  eval_strategy = 'epoch',
  save_strategy = 'epoch',
  load_best_model_at_end = True,
  report_to="none"
  )


  trainer = CustomTrainer(
  model = model,
  args = training_args,
  train_dataset = tokenized_data['train'],
  eval_dataset = tokenized_data['val'],
  tokenizer = tokenizer,
  data_collator = collate_fn,
  compute_metrics = compute_metrics,
  class_weights=class_weights,
  )

  train_result = trainer.train()
  logger.info(f"Model training done")
  output_dir = name_of_model

  trainer.save_model(output_dir)
  tokenizer.save_pretrained(output_dir)
  model.save_pretrained(output_dir)

  def generate_predictions(model,df_test):
    sentences = df_test.input.tolist()
    batch_size = 8
    all_outputs = []
    for i in tqdm(range(0, len(sentences), batch_size)):

      batch_sentences = sentences[i:i + batch_size]

      inputs = tokenizer(batch_sentences, return_tensors="pt",
      padding=True, truncation=True, max_length=1024)

      inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu')
      for k, v in inputs.items()}

      with torch.no_grad():
        outputs = model(**inputs)
        all_outputs.append(outputs['logits'])

    final_outputs = torch.cat(all_outputs, dim=0)
    df_test['predictions']=final_outputs.argmax(axis=1).cpu().numpy()
    
  logger.info(f"Starting predictions")
  generate_predictions(model,test_df)
  f1_score_ans = get_metrics_result(test_df)
  return f1_score_ans

#--------------------------------------------------------------------------
def finetune_model(r,lora_alpha,lora_dropout,epochs,batch_size,weight_decay,learning_rate):
  name_model = f"llama_r{r}_lora_alpha{lora_alpha}_lora_dropout{lora_dropout}_batch_size{batch_size}_weight_decay{weight_decay}_learning_rate{learning_rate}"
  f1_score_ans = model_train(r,lora_alpha,lora_dropout,epochs,batch_size,weight_decay,learning_rate,name_model)
  return f1_score_ans

#hyperparameters fixed for now
r = 16
lora_alpha = 16
lora_dropout = 0.05
epochs = 3
batch_size = 8
weight_decay = 0.01

#finetuning Learning rate
best_learning_rate = 0
best_f1 = 0
for learning_rate in [5e-5, 2e-5, 1e-5]: 
  print(f"Trying learning rate: {learning_rate}")
  logger.info(f"Trying learning rate: {learning_rate}")
  print(r,lora_alpha,lora_dropout,epochs,batch_size,weight_decay,learning_rate)
  f1_score_ans = finetune_model(r,lora_alpha,lora_dropout,epochs,batch_size,weight_decay,learning_rate)
  if f1_score_ans > best_f1:
    best_f1 = f1_score_ans
    best_learning_rate = learning_rate
print(f"Best learning rate: {best_learning_rate}")
logger.info(f"Best learning rate: {best_learning_rate}")

#finetuning LoRA Rank (Capacity Control)
learning_rate = best_learning_rate
lora_dropout = 0.05
epochs = 3
batch_size = 8
weight_decay = 0.01
best_r = 0
best_f1 = 0
for r in [8, 16, 32]:
  lora_alpha = r
  print(f"Trying LoRA Rank and alpha: {r} and {lora_alpha}")
  logger.info(f"Trying LoRA Rank: {r} and {lora_alpha}") 
  print(r,lora_alpha,lora_dropout,epochs,batch_size,weight_decay,learning_rate)
  f1_score_ans = finetune_model(r,lora_alpha,lora_dropout,epochs,batch_size,weight_decay,learning_rate)
  if f1_score_ans > best_f1:
    best_f1 = f1_score_ans
    best_r = r
print(f"Best LoRA Rank: {best_r}")
logger.info(f"Best LoRA Rank: {best_r}")

#finetuning LoRA Alpha (Scaling Fine-Tuning)
learning_rate = best_learning_rate
r = best_r
lora_dropout = 0.05
epochs = 3
batch_size = 8
weight_decay = 0.01

best_alpha = 0
best_f1 = 0
for lora_alpha in [r/2, r, 2*r]:
  print(f"Trying LoRA Rank and alpha: {r} and {lora_alpha}")
  logger.info(f"Trying LoRA Rank: {r} and {lora_alpha}") 
  print(r,lora_alpha,lora_dropout,epochs,batch_size,weight_decay,learning_rate)
  f1_score_ans = finetune_model(r,lora_alpha,lora_dropout,epochs,batch_size,weight_decay,learning_rate)
  if f1_score_ans > best_f1:
    best_f1 = f1_score_ans
    best_alpha = lora_alpha
print(f"Best LoRA alpha: {best_alpha}")
logger.info(f"Best LoRA alpha: {best_alpha}")

#finetuning LoRA Dropout and Weight Decay
learning_rate = best_learning_rate
r = best_r
lora_alpha = best_alpha
epochs = 3
batch_size = 8

best_dropout = 0
best_decay = 0
best_f1 = 0
for weight_decay in [0.0, 0.01]:
  for lora_dropout in [0.0, 0.05, 0.1]:
    print(f"Trying weight_decay: {weight_decay} and lora_dropout:{lora_dropout}")
    logger.info(f"Trying weight_decay: {weight_decay} and lora_dropout:{lora_dropout}") 
    print(r,lora_alpha,lora_dropout,epochs,batch_size,weight_decay,learning_rate)
    f1_score_ans = finetune_model(r,lora_alpha,lora_dropout,epochs,batch_size,weight_decay,learning_rate)
    if f1_score_ans > best_f1:
      best_f1 = f1_score_ans
      best_dropout = lora_dropout
      best_decay = weight_decay
print(f"Best weight_decay: {best_decay} and best lora_dropout:{best_dropout}")
logger.info(f"Best weight_decay: {best_decay} and best lora_dropout:{best_dropout}")

logger.info(f"Best learning rate: {best_learning_rate} Best LoRA Rank: {best_r} Best LoRA alpha: {best_alpha} Best weight_decay: {best_decay} and best lora_dropout:{best_dropout}")
print(f"Best learning rate: {best_learning_rate} Best LoRA Rank: {best_r} Best LoRA alpha: {best_alpha} Best weight_decay: {best_decay} and best lora_dropout:{best_dropout}")
