from unsloth import FastLanguageModel
import torch
import csv
import transformers
from unsloth.trainer import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import os
import re
import ast

from make_dataset import make_dataset_MSS_to_mol, make_dataset_mol_to_odor

def to_text(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return {"text": text}

# More models at https://huggingface.co/unsloth
### モデルのダウンロード
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit", # Change any model you want to use
    max_seq_length = 1024,
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # A bit more accurate, uses 2x memory
    full_finetuning = False, # Falseのままで
)
 
model=FastLanguageModel.get_peft_model(model) 
system='''You are an assistant that performs three types of tasks:
1. MSS_TO_MOLECULE:
   - Input: MSS sensor data and receptor type. - Output: The predicted molecule information, such as SMILES and Functional groups.
2. MOLECULE_TO_ODOR:
   - Input: Molecular information. - Output: Describe the odor of the molecule in three words or fewer.
3. MSS_TO_ODOR:
   This task is a combination of Task 1 and Task 2.
   - Input: MSS sensor data and receptor type. - Output: Describe the odor of the molecule in three words or fewer.
The user will *always* specify the task with:
"Task: MSS_TO_MOLECULE" or "Task: MOLECULE_TO_ODOR". or "Task: MSS_TO_ODOR"
Follow these rules:
- Always output ONLY the required fields for the task. - Do not include explanations. - Do not add extra text.'''
user1="Task: MSS_TO_MOLECULE\n"
user2="Task: MOLECULE_TO_ODOR\n"
train_dataset=[]
vali_dataset=[]
MSS_folder=""
odor_folder=""
MSS_vali=["Sample E","Sample F", "Sample G"]
odor_vali=["Sample S","Sample T","Sample R"]

make_dataset_MSS_to_mol(MSS_folder,train_dataset,vali_dataset,system,user1,MSS_vali)
make_dataset_mol_to_odor(odor_folder,train_dataset,vali_dataset,system,user2,odor_vali)

train_raw_ds=Dataset.from_list(train_dataset)
vali_raw_ds=Dataset.from_list(vali_dataset)

train_ds = train_raw_ds.map(to_text,remove_columns=["messages"])
vali_ds = vali_raw_ds.map(to_text,remove_columns=["messages"])

training_args = TrainingArguments(
    output_dir="./log_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # バッチサイズの実質的増加
    num_train_epochs=6,  
    logging_steps=50,  # ロギングの頻度
    save_strategy="epoch",           # 各エポックで保存
    eval_strategy="epoch",    # 各エポックで自動評価
    learning_rate=2e-5,  # 学習率
    bf16=True,# 混合精度
    optim="adamw_8bit",  # bitsandbytes を使った効率的な最適化
    lr_scheduler_type="linear",  # 学習率スケジュール
    load_best_model_at_end=True,     # 最良のval_lossモデルを最後に復元
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=vali_ds,
        #max_seq_length=512,
        tokenizer=tokenizer,
        dataset_text_field="text",
        packing=False,
        args=training_args,
        )
trainer.train()
tokenizer.save_pretrained("./output_finetune")
model.save_pretrained("./output_finetune")
