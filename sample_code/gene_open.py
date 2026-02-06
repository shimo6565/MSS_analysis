from unsloth import FastLanguageModel
import torch
from peft import PeftModel
import argparse

base_model="unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
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


def make_prompt(system,user,assistant):
    template={"messages":
            [{"role": "system", "content": system},
             {"role": "user", "content": user},
             {"role": "assistant", "content": assistant}]}
    return template

def generate_MSS_to_odor(txt_path,model,tokenizer):  
    with open(txt_path,'r',encoding='utf-8') as f:
            line = f.readline().strip()
            prompt1=make_prompt(system,line,"")
            text1=tokenizer.apply_chat_template(prompt1["messages"], tokenize=False,add_generation_prompt=True,)
    inputs = tokenizer(text=[text1], return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512, 
            do_sample=True, 
            temperature=0.1, # ランダム性の強さ(低いほど決定的)
            top_p=0.9, # 確率制限
            eos_token_id=tokenizer.eos_token_id,)    
    for out in outputs:
        output_text = tokenizer.decode(out, skip_special_tokens=True)
        print(output_text)
    return

def steps_generate(adaptor,path):###MSS→分子の出力用
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model, 
    max_seq_length = 1024,
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # A bit more accurate, uses 2x memory
    full_finetuning = False, 
    )
    model = PeftModel.from_pretrained(model, adaptor)
    model.eval()

    generate_MSS_to_odor(path,model,tokenizer)
    return


def main():
    parser = argparse.ArgumentParser(
        description="MSS odor prediction demo"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to MSS input text file",
    )
    args = parser.parse_args()
    steps_generate(args.adapter_path, args.data_path)



if __name__=="__main__":
    main()
