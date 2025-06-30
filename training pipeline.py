# install the libs: pip install datasets accelerate bitsandbytes peft
import torch, pandas as pd, numpy as np
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          BitsAndBytesConfig, DataCollatorForSeq2Seq,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate, random

csv="/content/drive/MyDrive/Kenya Clinical Reasoning Challenge/clean_train.csv"
sv_path="/content/drive/MyDrive/Kenya Clinical Reasoning Challenge/flan_small"

bat_size=4
lean_rate=0.0003219422218
num_epoch=7
sch_type="cosine"

seed=42
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

df=pd.read_csv(csv).rename(columns=str.lower)
assert {'input','target'}.issubset(df.columns)
dataset=Dataset.from_pandas(df[['input','target']])
dataset=dataset.train_test_split(test_size=0.1,seed=seed)
datasets=DatasetDict(train=dataset['train'],validation=dataset['test'])

model_name="google/flan-t5-small"
tokenizer=AutoTokenizer.from_pretrained(model_name)
source=512
target=128
def tokenize(batch):
    model_in=tokenizer(batch['input'],max_length=source,truncation=True)
    with tokenizer.as_target_tokenizer():
        labels=tokenizer(batch['target'],max_length=target,truncation=True)
    model_in['labels']=labels['input_ids']
    return model_in

tokenized_ds=datasets.map(tokenize,batched=True,remove_columns=datasets['train'].column_names)

bnb_cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
base_model=AutoModelForSeq2SeqLM.from_pretrained(model_name,quantization_config=bnb_cfg,device_map="auto")
base_model=prepare_model_for_kbit_training(base_model)
lora_cfg=LoraConfig(r=16,lora_alpha=32,target_modules=["q","v","k","o"],lora_dropout=0.05,bias="none",task_type="SEQ_2_SEQ_LM")
model=get_peft_model(base_model,lora_cfg)
model.print_trainable_parameters()

args=TrainingArguments(output_dir="/tmp/ignore_checkpoints",per_device_train_batch_size=bat_size,per_device_eval_batch_size=bat_size,learning_rate=lean_rate,num_train_epochs=num_epoch,lr_scheduler_type=sch_type,eval_strategy="epoch",save_strategy="no",logging_first_step=True,logging_steps=50,report_to="none",bf16=torch.cuda.is_bf16_supported(),dataloader_num_workers=2,gradient_accumulation_steps=1,gradient_checkpointing=True)

dara_collator=DataCollatorForSeq2Seq(tokenizer,model=model)

rouge=evaluate.load("rouge")
def compute_metrics(eval_pred):
    logits,labels=eval_pred
    preds=np.argmax(logits,axis=-1)
    decoded_preds=tokenizer.batch_decode(preds,skip_special_tokens=True)
    labels[labels==-100]=tokenizer.pad_token_id
    decoded_labels=tokenizer.batch_decode(labels,skip_special_tokens=True)
    score=rouge.compute(predictions=decoded_preds,references=decoded_labels,use_stemmer=True)
    return{"rougeL_f1":score["rougeL"]}

trainer=Trainer(model=model,args=args,train_dataset=tokenized_ds["train"],eval_dataset=tokenized_ds["validation"],data_collator=dara_collator,tokenizer=tokenizer)
trainer.train()

model.save_pretrained(sv_path)
tokenizer.save_pretrained(sv_path)
