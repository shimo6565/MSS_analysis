from unsloth import FastLanguageModel
import torch
import transformers
from unsloth.trainer import SFTTrainer
from transformers import TrainingArguments
from peft import PeftModel
from datasets import Dataset
import numpy as np
from dic import smi_to_label 
from make_dataset import make_prompt
# More models at https://huggingface.co/unsloth
### モデルのダウンロード
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit", # Change any model you want to use
    max_seq_length = 1024,
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # A bit more accurate, uses 2x memory
    full_finetuning = False, # Falseのままで
)
#model=FastLanguageModel.get_peft_model(model) 

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
user3="Task: MSS_TO_ODOR\n"

def generate_MSS_to_odor(name,con,model_epo,tokenizer,index):  #testデータの分子nameを濃度conについて一回出力　モデルmodelについて出力  ##MSS→分子よう
    prompt_list1=[]
    label=[]
    for rece in receptors:
        txt_path=folder+"/"+rece+"/"+name+con+".txt"
        with open(txt_path,'r',encoding='utf-8') as f:
                line = f.readline().strip()
                devide1=line.split("'concentration'")
                next2=user3+"'concentration'"+devide1[1].split("}")[0]
                prompt1=make_prompt(system,next2,"")
                text1=tokenizer.apply_chat_template(prompt1["messages"], tokenize=False,add_generation_prompt=True,)
                prompt_list1.append(text1)
    inputs = tokenizer(prompt_list1, return_tensors="pt", add_special_tokens=False, padding=True).to(model_epo.device)
    with torch.no_grad():
        outputs = model_epo.generate(
            **inputs,
            max_new_tokens=512, # 生成するトークンの最大長出力が途中で終わる場合は長くする
            do_sample=True, 
            temperature=0.1, # ランダム性の強さ(低いほど決定的)
            top_p=0.9, # 確率制限
            eos_token_id=tokenizer.eos_token_id,)    
    for out in outputs:
        output_text = tokenizer.decode(out, skip_special_tokens=True)
        answer=output_text.split("model")[2].strip()
        print(answer)
    return

def steps_generate(adaptor):###MSS→分子の出力用
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit", # Change any model you want to use
    max_seq_length = 1024,
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # A bit more accurate, uses 2x memory
    full_finetuning = False, # Falseのままで
    )
    model = PeftModel.from_pretrained(model, adaptor)
    model.eval()
    
    for i in range(6):
        mol=(test3_5)[i]
        generate_MSS_to_odor(mol,"10%",model,tokenizer,i)
    return